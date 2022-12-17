from enum import Enum


class ListedEnum(Enum):
    """
    Provides a simple way to list the values and names of its members.
    """

    @classmethod
    def values(cls):
        return [member.value for member in cls]

    @classmethod
    def names(cls):
        return [member.name for member in cls]


class LabelColors(ListedEnum):
    green = (89, 225, 50)
    red = (225, 50, 89)
    purple = (186, 50, 225)
    cyan = (50, 225, 186)
