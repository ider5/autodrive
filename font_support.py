from autodrive import i18n
from autodrive.i18n import set_chinese_font, use_english_labels

__all__ = ["set_chinese_font", "use_english_labels", "labels"]


def __getattr__(name):
    if name == "labels":
        return i18n.labels
    raise AttributeError(name)
