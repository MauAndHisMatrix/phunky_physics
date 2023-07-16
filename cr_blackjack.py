# MILESTONE PROJECT 2 ######################################################################

import random
import sys
from collections import namedtuple
from typing import List
from functools import total_ordering

Card = namedtuple('Card', ['suit', 'rank', 'value'])

class CardDeck:
    def __init__(self):
        self._suits = "Hearts Diamonds Spades Clubs".split()
        self._ranks = [str(x) for x in range(2, 11)] + list('JQKA')
        self._ranks_dict = dict((rank, i + 2) if i < 9 else (rank, 10) for i, rank in enumerate(self._ranks))
        self._ranks_dict['A'] = 11

        self._deck = [Card(suit=s, rank=r, value=v) for s in self._suits for r, v in self._ranks_dict.items()]

    def __getitem__(self, card):
        return self._deck[card]

    def __len__(self):
        return len(self._deck)

    def shuffle(self):
        random.shuffle(self._deck)

    def pick_cards(self, no_of_cards: int) -> list:
        return [self._deck.pop() for _ in range(no_of_cards)]


@total_ordering
class CardHand:
    def __init__(self):
        self._hand = []
        self.hand_value = 0
        self._ace_count = 0

    def __getitem__(self, card):
        return self._hand[card]

    def __repr__(self):
        return f'{self._hand}'

    def __eq__(self, other):
        if hasattr(other, 'hand_value'):
            return self.hand_value == other.hand_value
        return self.hand_value == other

    def __ge__(self, other):
        if hasattr(other, 'hand_value'):
            return self.hand_value >= other.hand_value
        return self.hand_value >= other

    def add_cards(self, cards: List[Card]):
        for card in cards:
            self._hand.append(card)
            if card.rank == 'A':
                self._ace_count += 1
            else:
                self.hand_value += card.value
        
        self.evaluate()

    def evaluate(self):
        while self._ace_count and self.hand_value > 21:
            self.hand_value -= 10
            self._ace_count -= 1
            

@total_ordering
class PlayerCash:
    def __init__(self, currently_owned_dough: int):
        self._cash = currently_owned_dough

    def __repr__(self):
        return f'£{self._cash} in the bank.'

    def __eq__(self, foe):
        return self._cash == foe

    def __ge__(self, foe):
        return self._cash >= foe

    def deduct(self, bet: int):
        if bet > self._cash:
            raise ValueError
        else:
            self._cash -= bet

        print(f"\nCURRENT LIFE SAVINGS: {self._cash}\n")

    def deposit(self, winnings: int):
        self._cash += winnings

        print(f"\nCURRENT LIFE SAVINGS: {self._cash}\n")


def blackjack(money_in_bank: PlayerCash):
    deck = CardDeck()
    dealer = CardHand()
    greedy_gambler = CardHand()
    life_savings = money_in_bank
    print("---------------------------------------------------------------------")
    print("\nWelcome to Blackjack you gambling bambino, let's rock the Kasbah!\n")
    print("---------------------------------------------------------------------")

    while True:
        bet = int(input("What is your initial bet you greedy bugger?\n"))
        try:
            life_savings.deduct(bet)
        except ValueError:
            if life_savings <= 0:
                print("What on Earth are you playing at son, you're drier than your mother's teets!\n")
                sys.exit(0)
            print("Nice try buddy, unfortunately you ain't got that lying around. Try again.\n")
            continue
        else:
            print("Congrats on performing some complex mathematics.\n")
            break

    deck.shuffle()
    greedy_gambler.add_cards(deck.pick_cards(2))
    print("--------------------------------------------------------------")
    print(f"\nMademoiselle, these are your cards: {greedy_gambler}\n")

    dealer.add_cards(deck.pick_cards(2))
    print(f"My cards are: {dealer[0]} and ???????   \n")
    print("--------------------------------------------------------------")

    while True:

        hit = input("\nMademoiselle, do you wish to hit?\n")

        if hit.lower() == 'yes':
            greedy_gambler.add_cards(deck.pick_cards(1))
            print(f"\nMademoiselle, these are your cards now: {greedy_gambler}\n")

            if greedy_gambler > 21:
                print("You've gone bust, fool.")
                print(f"Your losses are {bet}")
                break

        else:
            print("\nThe greedy bugger stands!\n")
            print(f'My hand as of now: {dealer}\n')
            while dealer < 17:
                print("Taking a hit myself...")
                dealer.add_cards(deck.pick_cards(1))
                print(f'My hand as of now: {dealer}\n')
            if dealer > 21 or greedy_gambler > dealer:
                print("Congratulations fool, seems like luck can go a long way...")
                print(f"Your winnings are {bet * 5}")
                life_savings.deposit(bet * 6)
                break
            elif greedy_gambler == dealer:
                print("You nearly won, so I won't bleed you dry.")
                life_savings.deposit(bet)
                break
            else:
                print("Unlucky chief, never bet against The House!")
                print(f"Your losses are {bet}")
                break
    
    print("And that's Blackjack, hope you learned some lessons and lost some money.")

    return life_savings

def decision_time():
    choice = input("\n\nSo you've been for one rodeo, fancy another?\n\
                    \n'a' = You bet, can do this all day!\
                    \n'b' = Nah, my wife is gonna kill me...")
    return choice


if __name__ == '__main__':
    print("\n\n-----------------------------------------------------------------------")
    print('-------------------WELCOME TO THE RISING SUN CASINO--------------------')
    print("-----------------------------------------------------------------------\n\n")
    cash = int(input("How much dough have you brought to the kitchen today?\n"))
    life_savings = PlayerCash(cash)
    choice = 'a'
    while choice == 'a':
        life_savings = blackjack(life_savings)
        choice = decision_time()
