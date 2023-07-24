# -*- coding: utf-8 -*-

"""Three Door Problem."""

# TODO: more doors, more fun!

import json
import random
from dataclasses import dataclass, asdict

from ..base.timer import timeit


@dataclass
class Door:
    """Door."""

    index: int
    prize: bool
    select: bool


class DoorsBase:
    """Abstract Multi-Doors Problem Solver."""

    def __init__(self, debug: bool = True) -> None:
        """Init."""
        self.debug = debug

    def prepare(self, number: int = 3) -> list[Door]:
        """Prepare Doors."""
        doors= [
            Door(index=index, prize=False, select=False)
            for index in range(number)
        ]
        door = random.choice(doors)
        door.prize = True
        door = random.choice(doors)
        door.select = True
        if self.debug:
            print("---doors.start---")
            self.show(doors)
            print("---doors.start---")
        return doors

    def show(self, doors: list[Door]) -> None:
        """Show doors."""
        data = [asdict(door) for door in doors]
        print(json.dumps(data, indent=2))
        print()

    def get_select(self, doors: list[Door]) -> Door:
        """Get right door."""
        return next(door for door in doors if door.select==True)

    def get_wrong(self, doors: list[Door]) -> Door:
        """Get wrong door."""
        random.shuffle(doors)
        return next(door for door in doors if door.prize==False and door.select==False)

    def drop_wrong(self, doors: list[Door]) -> list[Door]:
        """Expose one none-prize none-select door."""
        wrong = self.get_wrong(doors)
        remain = [door for door in doors if door != wrong]
        if self.debug:
            print("---remain.start---")
            self.show(remain)
            print("---remain.end---")
        return remain

    def is_right(self, doors: list[Door]) -> bool:
        """Check if select right prize."""
        return any(door for door in doors if door.prize==True and door.select==True)

    def strategy_change(self, doors: list[Door]) -> bool:
        """Strategy to change select after expose one none-prize door."""
        flag_last = False
        while len(doors) >= 3:
            doors = self.drop_wrong(doors)

            if len(doors) == 2 and flag_last == True:
                break

            # change select
            selected = self.get_select(doors)
            others = [door for door in doors if door != selected]
            selected.select = False
            random.choice(others).select = True
            # others.append(selected)
            # doors = others
            doors = others + [selected]

            if len(doors) == 2:
                flag_last = True

        return self.is_right(doors)

    def strategy_no_change(self, doors: list[Door]) -> bool:
        """Strategy no change select after expose one none-prize door."""
        while len(doors) > 2:
            doors = self.drop_wrong(doors)
        return self.is_right(doors)


class MultiDoors(DoorsBase):
    """Multi-Doors Problem Solver."""

    def __init__(self, debug: bool = True) -> None:
        super().__init__(debug)
        """Init."""

    def get_prob(self, change: bool, number:int = 3, times: int = 10) -> float:
        """Run to Get probability."""
        success = 0
        for _ in range(times):
            doors = self.prepare(number=number)
            if change:
                result = self.strategy_change(doors=doors)
            else:
                result = self.strategy_no_change(doors=doors)
            success += int(result)
            if self.debug:
                break
        return success / times

    @timeit
    def run(self) -> None:
        """Run."""
        self.debug = False
        number = 10
        times = 100000
        prob_change = self.get_prob(change=True, number=number, times=times)
        prob_no_change = self.get_prob(change=False, number=number, times=times)
        print(f"prob_change = {prob_change}")
        print(f"prob_no_change = {prob_no_change}")


if __name__ == "__main__":
     MultiDoors().run()