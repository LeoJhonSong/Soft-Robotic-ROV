from asciimatics.screen import Screen
import rov


class Controller():
    def __init__(self, robot: rov.Rov, screen: Screen) -> None:
        self.screen = screen
        self.events = {
            ord(' '): self.stop,
        }

    def key_check(self):
        key = self.screen.get_key()
        if key == ord('q'):
            return True
        else:
            if key in self.events:
                self.events[key]()
            return False

    def printScreen(self):
        # TODO: ui待设计
        # https://asciimatics.readthedocs.io/en/stable/asciimatics.html#asciimatics.screen.Screen.print_at
        self.screen.clear()
        self.screen.print_at('aaa', 0, 0)
        self.screen.refresh()

    def stop(self):
        pass
