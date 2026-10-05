# -*- coding: utf-8 -*-
"""跨聚落难民安置测试：外部聚落申请（每日轮换公告板）→ 地堡审核（批准/回绝）
→ 入境检疫（按人日消耗检疫物资、医护筛查潜藏疫病）→ 检疫期满接纳与岗位分配
（健康者入列、确诊者立为隔离病例），联动医疗病例、资源消耗、医疗危机与信誉结算；
并覆盖互斥锁、检疫难民口径、幂等回放、检疫病亡与终局安全收敛。"""
import json

import pytest
from sqlalchemy import text

from app.core.database import Base, engine, SessionLocal
from app.models import GameSession, Facility
from app.services.engine import (
    BunkerEngine,
    BunkerEngineError,
    BunkerEngineConflict,
    MED_ISOLATED,
    REFUGEE_QUARANTINE_DAYS,
    REFUGEE_QUARANTINE_COST,
    REFUGEE_INTAKE_COST,
    REFUGEE_REP_HEALTHY,
    REFUGEE_REP_REJECT,
    REFUGEE_REP_DEATH,
    REFUGEE_CASE_CRISIS,
    FOOD, WATER, POWER, OXY,
    FACILITY_ZH,
)
from app.core.migration import reconcile_old_saves
from tests.test_engine import make_session, FixedRand


@pytest.fixture()
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    s = SessionLocal()
    yield s
    s.close()
    Base.metadata.drop_all(bind=engine)


class RefugeeRand:
    """可脚本化随机：依次弹出 random() 值（检疫筛查掷骰），choice 取首项。

    默认 0.9：不触发地堡危机（0.9 > 0.45）、检疫筛查不命中（0.9 >= 检出率）。
    """

    def __init__(self, random_vals=()):
        self.vals = list(random_vals)

    def random(self):
        return self.vals.pop(0) if self.vals else 0.9

    def choice(self, seq):
        return seq[0]


def make_refugee_session(db, residents=3, resources=None, clinic=True):
    gs = make_session(db, residents=residents, resources=resources)
    if clinic:
        db.add(Facility(
            session_id=gs.id, name=FACILITY_ZH["clinic"], category="clinic",
            level=2, status="active", built_day=1,
        ))
        db.commit()
        db.refresh(gs)
    return gs


def first_petition(eng):
    return eng.refugee_board()[0]


def approve(eng, petition, jobs=None):
    jobs = jobs or ["general"] * petition["head"]
    return eng.review_refugee_petition(petition["id"], True, jobs=jobs)


def _write_people(gs, people):
    pet = dict(gs.refugee_petition)
    pet["people"] = people
    gs.refugee_petition = pet


def force_infected(gs, *indexes, health=None):
    """直接改写检疫快照，把指定难民标记为潜藏感染者（绕过公告板的信息隐藏）。"""
    target_ids = {gs.refugee_petition["people"][i]["resident_id"] for i in indexes}
    residents = {r.id: r for r in gs.residents}
    people = []
    for p in gs.refugee_petition["people"]:
        p = dict(p)
        if p["resident_id"] in target_ids:
            p["infected"] = True
            if health is not None:
                p["health"] = health
                residents[p["resident_id"]].health = health
        people.append(p)
    _write_people(gs, people)


def force_healthy(gs):
    """把检疫快照中的全部难民改为健康（用于确定走"全员接纳"路径）。"""
    residents = {r.id: r for r in gs.residents}
    people = []
    for p in gs.refugee_petition["people"]:
        p = dict(p)
        p["infected"] = False
        p["health"] = 80.0
        residents[p["resident_id"]].health = 80.0
        people.append(p)
    _write_people(gs, people)


def advance_quarantine(eng, days=REFUGEE_QUARANTINE_DAYS):
    for _ in range(days):
        eng.advance_day()


# ---- 公告板 ----

def test_board_is_deterministic_within_day(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    b1 = eng.refugee_board()
    b2 = eng.refugee_board()
    assert [p["id"] for p in b1] == [p["id"] for p in b2]
    assert len(b1) == 2
    assert all(1 <= p["head"] <= 3 for p in b1)


def test_board_rotates_across_days(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    day1 = {p["id"] for p in eng.refugee_board()}
    gs.day = 9
    day9 = {p["id"] for p in eng.refugee_board()}
    assert day1.isdisjoint(day9)


def test_public_board_never_exposes_infection(db):
    """公告视图必须剥离潜藏感染标记与确切健康，玩家无法预知谁带疫。"""
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    found_infected = any(
        p.get("summary", "").startswith("0") for p in eng.refugee_board()
    )
    assert not found_infected
    for p in eng.refugee_board_public():
        assert "infected" not in p
        assert "health" not in p
        assert set(p["people"][0].keys()) == {"name", "age"}


def test_expired_petition_cannot_be_reviewed(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    petition = first_petition(eng)
    gs.day += 1  # 跨天后公告板轮换，原申请失效
    with pytest.raises(BunkerEngineError):
        approve(eng, petition)


# ---- 审核 ----

def test_approve_validates_jobs(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    petition = first_petition(eng)
    with pytest.raises(BunkerEngineError):
        eng.review_refugee_petition(petition["id"], True, jobs=[])
    with pytest.raises(BunkerEngineError):
        eng.review_refugee_petition(
            petition["id"], True, jobs=["general"] * (petition["head"] + 1)
        )
    with pytest.raises(BunkerEngineError):
        bad = ["general"] * petition["head"]
        bad[0] = "king"
        eng.review_refugee_petition(petition["id"], True, jobs=bad)
    assert gs.refugee_petition is None


def test_approve_requires_intake_supplies(db):
    gs = make_refugee_session(
        db, resources={FOOD: 0, WATER: 300, POWER: 300, OXY: 300}
    )
    eng = BunkerEngine(db, gs, rand=FixedRand())
    petition = first_petition(eng)
    with pytest.raises(BunkerEngineError):
        approve(eng, petition)
    assert gs.refugee_petition is None


def test_approve_creates_quarantined_residents_and_freezes_intake(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    petition = first_petition(eng)
    jobs = ["engineer", "medic", "general", "farmer"][: petition["head"]]
    before = dict(gs.resources)
    detail, replayed = eng.review_refugee_petition(petition["id"], True, jobs=jobs)
    assert replayed is False
    # 洗消安置物资按人一次性扣除
    for k in (FOOD, WATER, POWER):
        assert gs.resources[k] == round(
            before[k] - REFUGEE_INTAKE_COST[k] * petition["head"], 1
        )
    # 难民即时建档为居民、人口计入，但在检疫区不属在堡人口/劳动力
    assert gs.survivors == 3 + petition["head"]
    qids = eng._quarantine_resident_ids()
    assert len(qids) == petition["head"]
    assert eng._in_bunker_count() == 3
    assert not (qids & {r.id for r in eng._in_bunker_residents()})
    # 岗位已随申请绑定（接纳前不可调岗）
    for p, job in zip(gs.refugee_petition["people"], jobs):
        resident = next(r for r in gs.residents if r.id == p["resident_id"])
        assert resident.job == job


def test_approve_is_idempotent_on_double_submit(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    petition = first_petition(eng)
    approve(eng, petition)
    survivors_after = gs.survivors
    # 同一份申请连点：安全回放，不重复建档/扣费
    detail, replayed = approve(eng, petition)
    assert replayed is True
    assert gs.survivors == survivors_after


def test_only_one_petition_at_a_time(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    approve(eng, first_petition(eng))
    other = eng.refugee_board()[1]
    with pytest.raises(BunkerEngineError):
        approve(eng, other)


def test_reject_costs_reputation_and_closes(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    petition = first_petition(eng)
    rep_before = gs.reputation
    detail, replayed = eng.review_refugee_petition(petition["id"], False)
    assert replayed is False
    assert gs.reputation == rep_before + REFUGEE_REP_REJECT
    assert gs.refugee_petition is None
    assert gs.survivors == 3
    # 连点回绝：幂等回放，不重复扣信誉
    _, replayed2 = eng.review_refugee_petition(petition["id"], False)
    assert replayed2 is True
    assert gs.reputation == rep_before + REFUGEE_REP_REJECT


def test_concurrent_losing_review_reconciles_as_replay(db):
    """并发落败（版本冲突）后核对：同一次审核安全回放，否则 409。"""
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    petition = first_petition(eng)
    approve(eng, petition)
    detail, replayed = eng.reconcile_stale_refugee(
        petition["id"], True, jobs=["general"] * petition["head"]
    )
    assert replayed is True
    with pytest.raises(BunkerEngineConflict):
        eng.reconcile_stale_refugee(petition["id"], False)


# ---- 互斥：检疫难民与离堡任务/经营动作 ----

def test_missions_excluded_while_petition_quarantining(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    approve(eng, first_petition(eng))
    # 检疫中：探索队/贸易/援助均不可发起（同一时间只办理一宗对外事务）
    with pytest.raises(BunkerEngineError):
        eng.send_expedition([gs.residents[0].id], {FOOD: 5, WATER: 5})
    offer = eng.trade_market()[0]
    with pytest.raises(BunkerEngineError):
        eng.apply_trade(offer["id"], [gs.residents[1].id])


def test_quarantined_resident_cannot_join_mission_or_change_job(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    approve(eng, first_petition(eng))
    qid = next(iter(eng._quarantine_resident_ids()))
    with pytest.raises(BunkerEngineError):
        eng.set_job(qid, "engineer")
    with pytest.raises(BunkerEngineError):
        eng.send_expedition([qid], {FOOD: 5, WATER: 5})
    with pytest.raises(BunkerEngineError):
        eng.register_case(qid)
    with pytest.raises(BunkerEngineError):
        eng.admit_case(qid, MED_ISOLATED)


def test_cannot_approve_while_expedition_away(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    # 先派探索队，再尝试审核难民申请：互斥拒绝
    eng.send_expedition([gs.residents[0].id], {FOOD: 10, WATER: 10})
    petition = first_petition(eng)
    with pytest.raises(BunkerEngineError):
        approve(eng, petition)


# ---- 检疫每日结算 ----

def test_quarantine_consumes_daily_supplies_per_head(db):
    gs = make_refugee_session(db)
    # 关停全部产出设施：当日资源变化只含在堡口粮 + 检疫区消耗，便于精确核对
    for f in gs.facilities:
        f.status = "offline"
    eng = BunkerEngine(db, gs, rand=FixedRand())
    petition = first_petition(eng)
    approve(eng, petition)
    n = petition["head"]
    before = dict(gs.resources)
    eng.advance_day()  # 检疫第 1 天
    from app.services.engine import BASE_CONSUME
    for k, per in REFUGEE_QUARANTINE_COST.items():
        bunker_consume = BASE_CONSUME[k] * 3  # 3 名在堡居民的人均口粮（检疫难民不吃地堡口粮）
        assert gs.resources[k] == round(before[k] - per * n - bunker_consume, 1)
    assert gs.refugee_petition["elapsed_days"] == 1


def test_quarantine_excludes_refugees_from_bunker_ration_and_production(db):
    """检疫难民不吃地堡口粮、不参与地堡生产（在堡人口口径不含检疫区）。"""
    gs = make_refugee_session(db, resources={FOOD: 500, WATER: 500, POWER: 500, OXY: 500})
    eng = BunkerEngine(db, gs, rand=FixedRand())
    petition = first_petition(eng)
    approve(eng, petition)
    # 推进当日：地堡消耗只按 3 名在堡居民计算（日志/产出可由在堡口径推出）
    assert eng._in_bunker_count() == 3
    eng.advance_day()
    # 检疫难民健康在检疫结算中变化（休整回复），但不吃地堡医疗舱外的口粮
    quarantined = eng._quarantine_residents()
    assert quarantined  # 仍在检疫区


def test_quarantine_shortage_damages_refugees(db):
    """检疫物资不足：当日停摆，难民健康与士气受损（但不殃及在堡居民）。"""
    gs = make_refugee_session(
        db, resources={FOOD: 500, WATER: 500, POWER: 500, OXY: 500}
    )
    eng = BunkerEngine(db, gs, rand=RefugeeRand())
    petition = first_petition(eng)
    approve(eng, petition)
    force_healthy(gs)
    # 检疫物资只留极少（低于当日按人需求）：第一个检疫日必然停摆；
    # 氧气维持安全线以上，避免全线枯竭触发终局
    gs.resources = {FOOD: 1, WATER: 1, POWER: 1, OXY: 60}
    health_before = {
        p["resident_id"]: p["health"] for p in gs.refugee_petition["people"]
    }
    eng.advance_day()
    for p in gs.refugee_petition["people"]:
        resident = next(r for r in gs.residents if r.id == p["resident_id"])
        assert resident.health < health_before[p["resident_id"]]


# ---- 检疫期满接纳 ----

def test_healthy_refugees_admitted_after_quarantine_with_jobs(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=RefugeeRand())  # 筛查不命中、不触发危机
    petition = first_petition(eng)
    jobs = ["engineer", "medic", "general", "farmer"][: petition["head"]]
    eng.review_refugee_petition(petition["id"], True, jobs=jobs)
    force_healthy(gs)  # 确定全员健康（公告板的潜藏感染不由测试预知）
    rep_before = gs.reputation
    advance_quarantine(eng)
    # 检疫期满：申请清除，全员接纳，无新增病例，信誉按健康人头提升
    assert gs.refugee_petition is None
    assert gs.survivors == 3 + petition["head"]
    assert gs.reputation == rep_before + REFUGEE_REP_HEALTHY * petition["head"]
    assert not (gs.medical_cases or [])
    # 岗位正式生效：难民按申请岗位入列，成为在堡劳动力
    assert eng._in_bunker_count() == 3 + petition["head"]
    admitted = gs.residents[3:]
    for r, job in zip(admitted, jobs):
        assert r.job == job and r.alive


def test_infected_refugees_detected_and_admitted_as_isolated_cases(db):
    gs = make_refugee_session(db)
    # 首日先过危机掷骰（0.99 不触发），之后检疫筛查掷骰也不命中，期满当日强制终检兜底
    eng = BunkerEngine(db, gs, rand=RefugeeRand([0.99] * 12))
    petition = first_petition(eng)
    approve(eng, petition)
    force_healthy(gs)
    force_infected(gs, *range(petition["head"]), health=55)
    crisis_before = eng.medical_crisis()
    advance_quarantine(eng)
    cases = gs.medical_cases or []
    assert len(cases) == petition["head"]
    assert all(c["status"] == MED_ISOLATED and c["infectious"] for c in cases)
    # 医疗危机按确诊人头抬升
    assert eng.medical_crisis() == min(
        100, crisis_before + REFUGEE_CASE_CRISIS * petition["head"]
    )
    # 病例绑定的居民均存活且已入列（病例随医疗救治中心继续救治）
    case_owners = {c["resident_id"] for c in cases}
    assert case_owners <= {r.id for r in gs.residents if r.alive}


def test_detected_refugee_flag_visible_during_quarantine(db):
    """医护筛查率随在岗医护提高；筛中后快照标记确诊，期满立为隔离病例。"""
    gs = make_refugee_session(db)
    # 给地堡配两名在任医护，筛查掷骰 0.40 < 0.45+0.10 命中
    gs.residents[0].job = "medic"
    gs.residents[1].job = "medic"
    # 首日：0.40 先触发并命中检疫筛查（感染难民的健康恶化掷骰），0.6 不触发危机
    eng = BunkerEngine(db, gs, rand=RefugeeRand([0.40, 0.6]))
    petition = first_petition(eng)
    approve(eng, petition)
    force_healthy(gs)
    force_infected(gs, 0, health=50)
    eng.advance_day()  # 检疫第 1 天：筛查命中
    detected = [p for p in gs.refugee_petition["people"] if p["detected"]]
    assert len(detected) == 1
    # 检疫期内尚未建立正式病例（检疫区不占救治床位）
    assert not (gs.medical_cases or [])
    advance_quarantine(eng, REFUGEE_QUARANTINE_DAYS - 1)
    assert any(c["infectious"] and c["status"] == MED_ISOLATED for c in (gs.medical_cases or []))


def test_refugee_death_during_quarantine_settled_at_admission(db):
    """潜藏感染者未被筛出且健康归零：检疫中病亡，人口扣减、接纳时信誉受罚。"""
    gs = make_refugee_session(db)
    # 0.99：不触发危机，检疫筛查也始终不命中（期满终检前）
    eng = BunkerEngine(db, gs, rand=RefugeeRand([0.99] * 12))
    petition = first_petition(eng)
    approve(eng, petition)
    force_healthy(gs)
    # 第一名带疫且已濒死（初始健康 6，每日恶化 5 → 第二日病亡）
    force_infected(gs, 0, health=6)
    rep_before = gs.reputation
    advance_quarantine(eng)
    # 一名病亡：总人口少 1，剩余健康者正常接纳
    assert gs.survivors == 3 + petition["head"] - 1
    assert gs.refugee_petition is None
    assert gs.reputation == rep_before + REFUGEE_REP_DEATH \
        + REFUGEE_REP_HEALTHY * (petition["head"] - 1)


def test_quarantine_does_not_block_daily_crisis(db):
    """检疫观察与地堡危机互不替代：危机仍按概率触发、在堡居民照常结算。"""
    gs = make_refugee_session(db)
    # 首个 0.1 触发地堡危机；其后筛查掷骰默认 0.9（不命中）
    eng = BunkerEngine(db, gs, rand=RefugeeRand([0.1]))
    petition = first_petition(eng)
    approve(eng, petition)
    force_healthy(gs)
    crisis = eng.advance_day()
    assert crisis is not None and crisis["event"]
    # 检疫天数照常累计，且危机挂起不影响检疫快照持久化
    assert gs.refugee_petition["elapsed_days"] == 1


# ---- 终局安全收敛 ----

def test_quarantine_force_admitted_on_target_day(db):
    """抵达目标日检疫未结束：强制安全接纳（无物资/信誉奖惩），不留僵尸申请。"""
    gs = make_refugee_session(db)
    gs.day = gs.target_day - 1
    # 首个 0.99 不触发危机（终局日），筛查掷骰同样不命中，由终检兜底
    eng = BunkerEngine(db, gs, rand=RefugeeRand([0.99] * 12))
    petition = first_petition(eng)
    approve(eng, petition)
    force_healthy(gs)
    force_infected(gs, *range(petition["head"]), health=55)
    rep_before = gs.reputation
    eng.advance_day()  # 抵达目标日：胜利终局
    assert gs.status == "win"
    assert gs.refugee_petition is None
    # 静默收敛：不发放接纳信誉奖励
    assert gs.reputation == rep_before
    # 但确诊事实不漏：终局病例履历完整
    cases = gs.medical_cases or []
    assert len(cases) == petition["head"]
    assert all(c["status"] == MED_ISOLATED for c in cases)


# ---- 旧存档迁移归一化 ----

def test_migration_clears_corrupt_refugee_petition(db):
    gs = make_refugee_session(db)
    db.commit()
    sid = gs.id
    # 结构残缺 / 居民编号悬空的检疫快照：归一化后清空，解除每日阶段死锁
    db.execute(
        text("UPDATE game_sessions SET refugee_petition = :p WHERE id = :sid"),
        {"p": json.dumps({
            "token": "t", "status": "quarantining",
            "people": [{"resident_id": 999999, "name": "虚"}],
        }), "sid": sid},
    )
    db.commit()
    n = reconcile_old_saves(engine)
    assert n == 1
    db.expire_all()
    fixed = db.get(GameSession, sid)
    assert fixed.refugee_petition is None
    eng = BunkerEngine(db, fixed, rand=FixedRand())
    assert eng.phase == "daily"
    eng.advance_day()
    db.commit()


def test_migration_clears_refugee_petition_on_ended_save(db):
    gs = make_refugee_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    approve(eng, first_petition(eng))
    gs.status = "over"
    db.commit()
    sid = gs.id
    reconcile_old_saves(engine)
    db.expire_all()
    assert db.get(GameSession, sid).refugee_petition is None
