from connect_x import ConnectX

def main():
    game = ConnectX()

    while (not game.is_done()):
        print(f"Valid positions: {game.valid_positions()}")
        col = int(input("Enter column: "))
        game.place_coin(col)
        print()
        print(game.board)
        print()
    
    print()
    print(game.board)

if __name__ == "__main__":
    main()
    
