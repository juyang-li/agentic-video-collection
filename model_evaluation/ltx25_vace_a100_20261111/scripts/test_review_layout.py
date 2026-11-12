import unittest
from html.parser import HTMLParser
from review_layout import show_condition_preview, task_navigation


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.targets = []
    def handle_starttag(self, tag, attrs):
        if tag == "a": self.targets.append(dict(attrs)["href"])


class LayoutTest(unittest.TestCase):
    def test_all_tasks_accessible_without_paging(self):
        parser = Links()
        parser.feed(task_navigation(["identity", "camera"], {"identity": "身份", "camera": "镜头"}))
        self.assertEqual(parser.targets, ["#top", "#task-identity", "#task-camera"])

    def test_only_two_global_condition_examples(self):
        for role in ("prepared_input", "overlay_video"):
            self.assertEqual([i for i in range(100) if show_condition_preview(role, i)], [0, 1])
        self.assertTrue(show_condition_preview("pose_video", 99))


if __name__ == "__main__": unittest.main()
