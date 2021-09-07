from asciimatics.screen import Screen
from asciimatics.event import KeyboardEvent
from . import auv


class Controller():
    def __init__(self, robot: auv.Auv, screen: Screen) -> None:
        self.screen = screen
        self.events = {
            ord(' '): self.stop,
        }

    def key_check(self):
        event = self.screen.get_event()
        if isinstance(event, KeyboardEvent):
            key = event.key_code
            if key == ord('q'):
                return True
            if key in self.events:
                self.events[key]()
        return False

    def printScreen(self):
        # https://asciimatics.readthedocs.io/en/stable/asciimatics.html#asciimatics.screen.Screen.print_at
        self.screen.clear()
        self.screen.refresh()

    def stop(self):
        pass
