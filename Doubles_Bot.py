import asyncio
import numpy as np
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import BattleOrder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env import ServerConfiguration


class SmartBot(Player):
    """Bot that chooses the move with highest type effectiveness against the current opponent."""

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # Singles: available_moves is a list; Doubles: list of two lists per active mon
        if battle.format_is_doubles:
            return self._choose_doubles_move(battle)
        return self._choose_singles_move(battle)

    def _choose_singles_move(self, battle: AbstractBattle) -> BattleOrder:
        moves = battle.available_moves
        opponent = battle.opponent_active_pokemon

        if not moves:
            return self.choose_random_move(battle)
        if opponent is None or opponent.fainted:
            return Player.create_order(moves[0])

        # Type effectiveness of each move against the opponent
        effectiveness = np.array([opponent.damage_multiplier(move) for move in moves])
        # Rank by effectiveness (highest first); break ties by index with stable sort
        ranks = np.argsort(-effectiveness, kind="stable")
        best_idx = int(ranks[0])
        return Player.create_order(moves[best_idx])

    def _choose_doubles_move(self, battle: AbstractBattle) -> BattleOrder:
        from poke_env.battle.double_battle import DoubleBattle
        from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder, DefaultBattleOrder

        battle = battle  # type: DoubleBattle
        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        orders = []
        for mon, move_list in zip(battle.active_pokemon, battle.available_moves):
            if not mon or mon.fainted:
                orders.append(PassBattleOrder())
                continue
            if not move_list:
                orders.append(DefaultBattleOrder())
                continue

            # Score each move by best type effectiveness against either opponent
            opponents = [p for p in battle.opponent_active_pokemon if p and not p.fainted]
            if not opponents:
                orders.append(Player.create_order(move_list[0]))
                continue

            effectiveness = np.array([
                max(opp.damage_multiplier(m) for opp in opponents)
                for m in move_list
            ])
            ranks = np.argsort(-effectiveness, kind="stable")
            best_move = move_list[int(ranks[0])]
            orders.append(Player.create_order(best_move))

        return DoubleBattleOrder(orders[0], orders[1])


async def main():
    # Connect to the stadium running in your other tab
    local_server = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000")

    # Create two bots to test the connection
    bot_1 = RandomPlayer(battle_format="gen9vgc2024regg", server_configuration=local_server)
    bot_2 = RandomPlayer(battle_format="gen9vgc2024regg", server_configuration=local_server)

    print("Testing connection... Battle starting!")
    await bot_1.battle_against(bot_2, n_battles=1)
    print("Success! The bots played a full game.")

if __name__ == "__main__":
    asyncio.run(main())