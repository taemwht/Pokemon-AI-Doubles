"""
AdolenceBot (upgraded policy / evaluation) + battle runner.

Debug AdolenceBot and match setup here. Shared doubles logic: ``bots.py``
(ToddlerBot, RandomBot, BaseDoubleBot, team paste, snapshots).
"""

from __future__ import annotations

import asyncio
import random
import string
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.teambuilder import ConstantTeambuilder

from bots import REGULATION_I_TEAM, BaseDoubleBot, ToddlerBot

__all__ = ["AdolenceBot", "main"]


class AdolenceBot(BaseDoubleBot):
    """Base doubles tactics + position evaluation (for future search / upgrades)."""

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self._log_opponent_hp_delta(battle)

        # Temporary sanity check — remove after Step 1 is confirmed working
        score = self.evaluate_position(battle)
        print(f"[EVAL] position_score={score:.4f}")

        if not hasattr(self, "_debug_done"):
            self._debug_battle_sides(battle)
            self._debug_done = True

        order = self._choose_doubles_move(battle)
        self._last_opp_active_hp = self._opponent_active_hp_snapshot(battle)
        return order

    def _debug_battle_sides(self, battle: AbstractBattle) -> None:
        print(f"[SIDES] type={type(battle).__name__}")
        print(f"[SIDES] active_pokemon={[getattr(m, 'species', '?') for m in (battle.active_pokemon or [])]}")
        print(f"[SIDES] opponent_active_pokemon={[getattr(m, 'species', '?') for m in (battle.opponent_active_pokemon or [])]}")
        print(f"[SIDES] team keys={[getattr(m, 'species', '?') for m in battle.team.values()]}")
        print(f"[SIDES] opponent_team keys={[getattr(m, 'species', '?') for m in (battle.opponent_team or {}).values()]}")

    def _get_opponent_active(self, battle: AbstractBattle) -> list:
        """
        Safely get opponent active mons by excluding our own team species
        from whatever poke-env returns as opponent_active_pokemon.
        Since both bots use the same team, we need slot-based disambiguation.
        Use the raw battle._opponent_active_pokemon if available.
        """
        # Our species set
        our_species = {
            str(getattr(m, "species", "") or "").lower()
            for m in (battle.active_pokemon or [])
            if m is not None
        }

        def _is_live_mon(m) -> bool:
            return m is not None and hasattr(m, "current_hp_fraction")

        # Try the internal attribute first (may mix non-Pokemon placeholders; skip those)
        raw = getattr(battle, "_opponent_active_pokemon", None)
        if raw is not None:
            result = []
            for m in raw:
                if not _is_live_mon(m):
                    continue
                if getattr(m, "fainted", False):
                    continue
                result.append(m)
            if result:
                return result

        # Fallback — filter opponent_active_pokemon by excluding our active mons
        result = []
        our_active_ids = {id(m) for m in (battle.active_pokemon or []) if _is_live_mon(m)}
        for m in (battle.opponent_active_pokemon or []):
            if not _is_live_mon(m):
                continue
            if id(m) in our_active_ids:
                continue
            if getattr(m, "fainted", False):
                continue
            result.append(m)
        return result

    def evaluate_position(self, battle: AbstractBattle) -> float:
        """
        Score the board from our perspective. Higher = better for us.
        Called both as a standalone heuristic AND by the search tree at leaf nodes.
        """
        score = 0.0

        # --- HP advantage ---
        for mon in (battle.active_pokemon or []):
            if mon and not getattr(mon, "fainted", False):
                score += mon.current_hp_fraction * 1.5

        for mon in battle.team.values():
            if not getattr(mon, "fainted", False) and not getattr(mon, "active", False):
                score += mon.current_hp_fraction * 0.8

        for mon in self._get_opponent_active(battle):
            if not getattr(mon, "fainted", False):
                score -= mon.current_hp_fraction * 1.5

        for mon in (battle.opponent_team.values() or []):
            if not getattr(mon, "fainted", False) and not getattr(mon, "active", False):
                score -= mon.current_hp_fraction * 0.8

        # --- Numbers advantage ---
        our_alive = sum(1 for m in battle.team.values() if not getattr(m, "fainted", False))
        opp_alive = sum(
            1
            for m in (battle.opponent_team.values() or [])
            if not getattr(m, "fainted", False)
        )
        score += (our_alive - opp_alive) * 1.0

        # --- Status advantage ---
        bad_statuses = {"slp", "frz"}
        ok_statuses = {"brn", "psn", "tox", "par"}

        for mon in self._get_opponent_active(battle):
            if mon is None:
                continue
            s = str(getattr(mon, "status", "") or "").lower()
            if any(b in s for b in bad_statuses):
                score += 0.6
            elif any(b in s for b in ok_statuses):
                score += 0.2

        for mon in (battle.active_pokemon or []):
            if mon is None:
                continue
            s = str(getattr(mon, "status", "") or "").lower()
            if any(b in s for b in bad_statuses):
                score -= 0.6
            elif any(b in s for b in ok_statuses):
                score -= 0.2

        # --- Field conditions ---
        try:
            fields = getattr(battle, "fields", {}) or {}
            tr_active = any("trick" in str(k).lower() for k in fields.keys())
            if tr_active:
                score += 0.3 if self._our_team_wants_trick_room(battle) else -0.3
        except Exception:
            pass

        try:
            our_side = getattr(battle, "side_conditions", {}) or {}
            opp_side = getattr(battle, "opponent_side_conditions", {}) or {}
            if any("tailwind" in str(k).lower() for k in our_side.keys()):
                score += 0.25
            if any("tailwind" in str(k).lower() for k in opp_side.keys()):
                score -= 0.25
        except Exception:
            pass

        return score

    def _our_team_wants_trick_room(self, battle: AbstractBattle) -> bool:
        """
        True if our currently active mons benefit from Trick Room.
        Checks active slot first — if Ursaluna or another slow mon is
        active, TR is worth setting regardless of the full roster.
        """
        slow_threshold = 80  # raised from 60 — catches more mid-speed mons too

        # Check active mons first — if any active mon is slow, TR is worth it
        active_slow = 0
        active_total = 0
        for mon in (battle.active_pokemon or []):
            if mon is None or getattr(mon, "fainted", False):
                continue
            base_spe = (getattr(mon, "base_stats", {}) or {}).get("spe", 999)
            active_total += 1
            if base_spe <= slow_threshold:
                active_slow += 1

        if active_total > 0 and active_slow >= 1:
            return True  # at least one active mon wants TR

        # Fall back to bench check — if slow mons are waiting to come in
        bench_slow = 0
        bench_total = 0
        for mon in battle.team.values():
            if getattr(mon, "fainted", False) or getattr(mon, "active", False):
                continue
            base_spe = (getattr(mon, "base_stats", {}) or {}).get("spe", 999)
            bench_total += 1
            if base_spe <= slow_threshold:
                bench_slow += 1

        return bench_total > 0 and (bench_slow / bench_total) >= 0.5

    def _score_move(self, damage_fraction: float, defender_hp_fraction: float) -> float:
        """
        KO priority scoring — overrides base class raw damage scoring.
        A KO scores ~10x higher than the best non-KO move.
        """
        if damage_fraction <= 0:
            return 0.0

        if damage_fraction >= defender_hp_fraction:
            return 10.0 + min(damage_fraction, 1.0)

        damage_ratio = damage_fraction / defender_hp_fraction
        return damage_ratio * 2.0

    def _score_utility_move(self, move, attacker, battle: AbstractBattle) -> float:
        move_id = str(getattr(move, "id", "") or "").lower().replace("-", "").replace(" ", "")
        attacker_name = str(getattr(attacker, "species", None) or "?").lower()

        if move_id == "trickroom":
            for mon in (battle.active_pokemon or []):
                if mon is None:
                    continue
                base_spe = (getattr(mon, "base_stats", {}) or {}).get("spe", 999)
                print(f"[TR ACTIVE DEBUG] species={getattr(mon, 'species', '?')} base_spe={base_spe}")

        if move_id == "protect":
            facing_ko = self._facing_likely_ko(attacker, battle)
            attacker_hp = getattr(attacker, "current_hp_fraction", 1.0)
            if facing_ko:
                print(f"[UTILITY] protect={attacker_name} reason=facing_ko hp={attacker_hp:.2f} score=8.0")
                return 8.0
            if attacker_hp <= 0.4:
                print(f"[UTILITY] protect={attacker_name} reason=low_hp hp={attacker_hp:.2f} score=6.0")
                return 6.0
            if attacker_hp <= 0.6:
                print(f"[UTILITY] protect={attacker_name} reason=mid_hp hp={attacker_hp:.2f} score=2.0")
                return 2.0
            if self._should_stall(battle):
                print(f"[UTILITY] protect={attacker_name} reason=stall hp={attacker_hp:.2f} score=3.0")
                return 3.0
            print(f"[UTILITY] protect={attacker_name} reason=none hp={attacker_hp:.2f} score=-1.0")
            return -1.0

        if move_id == "spore":
            # Get our own species names to filter them out
            our_species = {
                str(getattr(m, "species", "") or "").lower()
                for m in battle.team.values()
                if m is not None
            }

            for opp in self._get_opponent_active(battle):
                if opp is None or getattr(opp, "fainted", False):
                    continue
                opp_species = str(getattr(opp, "species", "") or "").lower()
                # Skip if this is actually one of our own mons
                if opp_species in our_species:
                    print(f"[SPORE DEBUG] skipping own mon species={opp_species}")
                    continue
                item = str(getattr(opp, "item", "") or "").lower().replace(" ", "")
                if item in ("lumberry", "safetygoggles"):
                    print(f"[UTILITY] spore blocked by item={item} on {opp_species}")
                    continue
                status = str(getattr(opp, "status", "") or "").lower()
                if "slp" in status:
                    print(f"[UTILITY] spore blocked already_asleep {opp_species}")
                    continue
                print(f"[UTILITY] spore={attacker_name} target={opp_species} score=7.0")
                return 7.0
            print(f"[UTILITY] spore={attacker_name} no_valid_target score=-1.0")
            return -1.0

        if move_id in ("ragepowder", "followme"):
            partner = self._get_partner(attacker, battle)
            if partner is None:
                print(f"[UTILITY] ragepowder={attacker_name} no_partner score=-1.0")
                return -1.0
            partner_hp = getattr(partner, "current_hp_fraction", 0.0)
            attacker_hp = getattr(attacker, "current_hp_fraction", 0.0)
            if partner_hp > attacker_hp + 0.3:
                print(
                    f"[UTILITY] ragepowder={attacker_name} partner={getattr(partner, 'species', '?')} "
                    f"partner_hp={partner_hp:.2f} our_hp={attacker_hp:.2f} score=6.0"
                )
                return 6.0
            print(f"[UTILITY] ragepowder={attacker_name} partner_not_worth_protecting score=-1.0")
            return -1.0

        if move_id == "wideguard":
            has_spread = self._opponent_has_spread_moves(battle)
            print(
                f"[UTILITY] wideguard={attacker_name} opp_has_spread={has_spread} "
                f"score={'5.0' if has_spread else '-1.0'}"
            )
            if has_spread:
                return 5.0
            return -1.0

        if move_id == "trickroom":
            tr_active = any(
                "trick" in str(k).lower()
                for k in (getattr(battle, "fields", {}) or {}).keys()
            )
            wants_tr = self._our_team_wants_trick_room(battle)
            if tr_active:
                print(f"[UTILITY] trickroom={attacker_name} already_active score=-1.0")
                return -1.0
            if wants_tr:
                print(f"[UTILITY] trickroom={attacker_name} setting_tr score=8.0")
                return 8.0
            print(f"[UTILITY] trickroom={attacker_name} team_doesnt_want_tr score=-1.0")
            return -1.0

        if move_id == "tailwind":
            our_side = getattr(battle, "side_conditions", {}) or {}
            if any("tailwind" in str(k).lower() for k in our_side.keys()):
                print(f"[UTILITY] tailwind={attacker_name} already_active score=-1.0")
                return -1.0
            print(f"[UTILITY] tailwind={attacker_name} setting score=5.0")
            return 5.0

        if move_id in ("snarl", "icywind"):
            print(f"[UTILITY] debuff={move_id} attacker={attacker_name} score=1.5")
            return 1.5

        print(f"[UTILITY] unknown={move_id} attacker={attacker_name} score=-1.0")
        return -1.0

    def _facing_likely_ko(self, attacker, battle: AbstractBattle) -> bool:
        attacker_hp = getattr(attacker, "current_hp_fraction", 1.0)
        for opp in self._get_opponent_active(battle):
            if opp is None or getattr(opp, "fainted", False):
                continue
            for move in (getattr(opp, "moves", {}) or {}).values():
                if getattr(move, "base_power", 0) == 0:
                    continue
                dmg = self._estimate_damage(move, opp, attacker, battle)
                if dmg >= attacker_hp:
                    return True
        return False

    def _should_stall(self, battle: AbstractBattle) -> bool:
        try:
            fields = getattr(battle, "fields", {}) or {}
            for k, v in fields.items():
                k_str = str(k).lower()
                if "trick" in k_str and isinstance(v, int) and v <= 2:
                    if not self._our_team_wants_trick_room(battle):
                        return True
        except Exception:
            pass
        return False

    def _get_partner(self, attacker, battle: AbstractBattle) -> object | None:
        for mon in (battle.active_pokemon or []):
            if mon is None or mon is attacker:
                continue
            if not getattr(mon, "fainted", False):
                return mon
        return None

    def _opponent_has_spread_moves(self, battle: AbstractBattle) -> bool:
        spread_ids = {
            "earthquake", "rockslide", "discharge", "surf",
            "heatwave", "blizzard", "bulldoze", "glaciallance",
            "hyperdrill", "expandingforce"
        }
        for opp in self._get_opponent_active(battle):
            if opp is None:
                continue
            for move in (getattr(opp, "moves", {}) or {}).values():
                mid = str(getattr(move, "id", "") or "").lower().replace("-", "")
                if mid in spread_ids:
                    return True
        return False


async def main():
    # Regulation I format; same team via ConstantTeambuilder — AdolenceBot vs ToddlerBot.
    vgc_format = "gen9vgc2026regi"
    team = ConstantTeambuilder(REGULATION_I_TEAM)
    suffix1 = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    suffix2 = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))

    bot_1 = AdolenceBot(
        battle_format=vgc_format,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        account_configuration=AccountConfiguration(f"AdolenceBot {suffix1}", ""),
    )
    bot_2 = ToddlerBot(
        battle_format=vgc_format,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        account_configuration=AccountConfiguration(f"ToddlerBot {suffix2}", ""),
    )

    total_battles = 1
    print(f"Running {total_battles} battles (AdolenceBot vs ToddlerBot)...")

    adolence_wins = 0
    toddler_wins = 0
    ties = 0

    results_path = Path(__file__).with_name("battle_results.csv")
    with results_path.open("w", encoding="utf-8") as f:
        f.write("battle_id,winner\n")

    for i in range(total_battles):
        await bot_1.battle_against(bot_2, n_battles=1)

        winner_label = "tie"
        bot_1.won = None
        finished = [b for b in bot_1.battles.values() if b.finished]
        if finished:
            latest = max(finished, key=lambda b: getattr(b, "battle_tag", ""))
            bot_1.won = latest.won
        if bot_1.won is True:
            winner_label = "AdolenceBot"
            adolence_wins += 1
        elif bot_1.won is False:
            winner_label = "ToddlerBot"
            toddler_wins += 1
        else:
            ties += 1

        with results_path.open("a", encoding="utf-8") as f:
            f.write(f"{i + 1},{winner_label}\n")

        print(f"Battle {i + 1}/{total_battles} done, winner={winner_label}")

        bot_1.reset_battles()
        bot_2.reset_battles()

    winrate_pct = 100.0 * adolence_wins / total_battles
    print("Done!")
    print(
        f"AdolenceBot winrate ({total_battles} battles): "
        f"{adolence_wins}W / {toddler_wins}L / {ties}T → "
        f"{winrate_pct:.2f}% wins (vs ToddlerBot)"
    )


if __name__ == "__main__":
    asyncio.run(main())
