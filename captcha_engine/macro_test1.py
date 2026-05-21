import time
import random
from playwright.sync_api import sync_playwright

URL = "http://localhost:5173/captcha"  # 너희 캡챠 테스트 페이지 주소로 수정

CANVAS_SELECTOR = "canvas"  # 캡챠 canvas selector로 수정


def human_like_move(page, start_x, start_y, end_x, end_y, steps=25):
    for i in range(steps):
        r = i / (steps - 1)

        # 직선 + 약간의 흔들림
        x = start_x + (end_x - start_x) * r + random.uniform(-3, 3)
        y = start_y + (end_y - start_y) * r + random.uniform(-3, 3)

        page.mouse.move(x, y)
        time.sleep(random.uniform(0.01, 0.04))


def grid_search_bot(page, canvas_box):
    """
    정답을 모르는 봇:
    손전등으로 화면을 지그재그 탐색하는 방식
    """
    left = canvas_box["x"]
    top = canvas_box["y"]
    width = canvas_box["width"]
    height = canvas_box["height"]

    step = random.randint(50, 80)

    current_x = left + 20
    current_y = top + 20

    page.mouse.move(current_x, current_y)

    y = top + 30
    row = 0

    while y < top + height - 30:
        if row % 2 == 0:
            x_values = range(int(left + 30), int(left + width - 30), step)
        else:
            x_values = range(int(left + width - 30), int(left + 30), -step)

        for x in x_values:
            human_like_move(page, current_x, current_y, x, y, steps=random.randint(8, 18))
            current_x, current_y = x, y
            time.sleep(random.uniform(0.05, 0.15))

        y += step
        row += 1


def random_click_bot(page, canvas_box):
    """
    정답을 모르는 봇:
    랜덤 탐색 후 랜덤 클릭
    """
    left = canvas_box["x"]
    top = canvas_box["y"]
    width = canvas_box["width"]
    height = canvas_box["height"]

    x = left + random.randint(20, int(width - 20))
    y = top + random.randint(20, int(height - 20))

    page.mouse.move(x, y)

    for _ in range(random.randint(20, 60)):
        nx = left + random.randint(10, int(width - 10))
        ny = top + random.randint(10, int(height - 10))

        human_like_move(page, x, y, nx, ny, steps=random.randint(5, 15))
        x, y = nx, ny
        time.sleep(random.uniform(0.03, 0.12))

    page.mouse.click(x, y)


def known_target_bot(page, canvas_box, target_x_norm, target_y_norm):
    """
    정답 좌표를 아는 봇:
    API/라벨이 노출된 경우를 가정한 공격
    """
    left = canvas_box["x"]
    top = canvas_box["y"]
    width = canvas_box["width"]
    height = canvas_box["height"]

    target_x = left + width * target_x_norm
    target_y = top + height * target_y_norm

    start_x = left + random.randint(10, int(width - 10))
    start_y = top + random.randint(10, int(height - 10))

    page.mouse.move(start_x, start_y)
    time.sleep(random.uniform(0.2, 0.8))

    human_like_move(
        page,
        start_x,
        start_y,
        target_x,
        target_y,
        steps=random.randint(15, 35)
    )

    time.sleep(random.uniform(0.1, 0.4))
    page.mouse.click(target_x, target_y)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=20)
        page = browser.new_page(viewport={"width": 1280, "height": 800})

        page.goto(URL)
        page.wait_for_selector(CANVAS_SELECTOR)

        canvas = page.locator(CANVAS_SELECTOR)
        box = canvas.bounding_box()

        if box is None:
            print("canvas를 찾지 못했습니다.")
            browser.close()
            return

        # 1. 랜덤 봇 테스트
        # random_click_bot(page, box)

        # 2. 그리드 탐색 봇 테스트
        grid_search_bot(page, box)

        # 3. 정답 좌표를 아는 봇 테스트
        # known_target_bot(page, box, target_x_norm=0.5, target_y_norm=0.3)

        time.sleep(3)
        browser.close()


if __name__ == "__main__":
    main()