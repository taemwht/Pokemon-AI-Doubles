import asyncio
from poke_env.player import RandomPlayer
from poke_env import ServerConfiguration

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