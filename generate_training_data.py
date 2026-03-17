"""
Generate a training dataset via self-play: SmartBot vs SmartBot.
Every turn, save the embed_battle vector. After each match, winner's turns → Target 1, loser's → 0.
Saves to processed_training_data.csv and battle_results.csv (who won each battle).
"""
import asyncio
import csv
import numpy as np

from poke_env.ps_client.account_configuration import AccountConfiguration

from Doubles_Bot import (
    SmartBot,
    ServerConfiguration,
    ConstantTeambuilder,
    REGULATION_F_TEAM,
)

OUTPUT_FILE = "processed_training_data.csv"
RESULTS_FILE = "battle_results.csv"
N_BATTLES = 200
# Use a format that allows our team (Lunala, Koraidon = restricted). Reg I allows 2 restricted.
VGC_FORMAT = "gen9vgc2026regi"


async def main():
    local_server = ServerConfiguration(
        "ws://localhost:8000/showdown/websocket",
        "http://localhost:8000",
    )
    vgc_format = VGC_FORMAT
    team = ConstantTeambuilder(REGULATION_F_TEAM)
    print(f"Format: {vgc_format}")
    print(f"Battles run on server: {local_server.websocket_url}")
    print(f"To watch live: open http://localhost:8000 in a browser and spectate the bot battles.\n")

    # Unique usernames so multiple runs don't conflict (e.g. "SmartBotTrain a1b2c")
    bot_1 = SmartBot(
        battle_format=vgc_format,
        server_configuration=local_server,
        team=team,
        account_configuration=AccountConfiguration.generate("SmartBotTrain", rand=True),
    )
    bot_2 = SmartBot(
        battle_format=vgc_format,
        server_configuration=local_server,
        team=team,
        account_configuration=AccountConfiguration.generate("SmartBotTrain", rand=True),
    )

    all_rows = []
    battle_results = []  # list of (battle_id, winner)

    print(f"Running {N_BATTLES} self-play battles (SmartBot vs SmartBot)...")
    for i in range(N_BATTLES):
        await bot_1.battle_against(bot_2, n_battles=1)

        # Who won: from bot_1's perspective (bot_2.won = opposite)
        for b in bot_1.battles.values():
            if b.finished:
                bot_1.won = b.won
                bot_2.won = not b.won
                break

        winner_name = bot_1.username if bot_1.won else bot_2.username
        battle_id = i + 1
        battle_results.append((battle_id, winner_name))
        print(f"Battle {battle_id}/{N_BATTLES}: {winner_name} won")

        # Winner's turns → Target 1, loser's turns → Target 0
        target_1 = 1 if bot_1.won else 0
        target_2 = 1 if bot_2.won else 0

        for vec, _ in bot_1.history:
            all_rows.append(np.concatenate([vec, np.array([target_1], dtype=np.float32)]))
        for vec, _ in bot_2.history:
            all_rows.append(np.concatenate([vec, np.array([target_2], dtype=np.float32)]))

        bot_1.history = []
        bot_2.history = []

        if (i + 1) % 20 == 0:
            print(f"  --- {i + 1}/{N_BATTLES} battles, {len(all_rows)} rows so far ---")

    # Save battle results CSV (who won each battle)
    with open(RESULTS_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["battle_id", "winner"])
        w.writerows(battle_results)
    print(f"\nSaved {len(battle_results)} results to {RESULTS_FILE}")

    if not all_rows:
        print("No training data collected.")
        return

    data = np.vstack(all_rows)
    n_features = data.shape[1] - 1
    header = ",".join(f"f{i}" for i in range(n_features)) + ",Target"
    np.savetxt(
        OUTPUT_FILE,
        data,
        delimiter=",",
        header=header,
        comments="",
    )
    print(f"Saved {len(all_rows)} rows to {OUTPUT_FILE} (shape {data.shape})")


if __name__ == "__main__":
    asyncio.run(main())
