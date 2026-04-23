import random

import pygame


WIDTH = 960
HEIGHT = 640
FPS = 120
BG = (18, 20, 24)
INVALID = (30, 34, 42)
VALID = (210, 220, 235)
CURSOR_FREE = (255, 210, 80)
CURSOR_SNAPPED = (110, 255, 160)
MODE_NEAREST = 1
MODE_CONTINUITY = 2


def draw_noisy_blob(surface, rng, rect):
    if rng.random() < 0.5:
        pygame.draw.ellipse(surface, VALID, rect)
    else:
        pygame.draw.rect(surface, VALID, rect, border_radius=rng.randint(8, 28))

    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2

    for _ in range(rng.randint(10, 22)):
        angle = rng.random() * 6.283185307179586
        reach_x = max(8, w // 2 + rng.randint(-10, 16))
        reach_y = max(8, h // 2 + rng.randint(-10, 16))
        px = int(cx + reach_x * pygame.math.Vector2(1, 0).rotate_rad(angle).x)
        py = int(cy + reach_y * pygame.math.Vector2(1, 0).rotate_rad(angle).y)
        radius = rng.randint(4, 12)
        pygame.draw.circle(surface, VALID, (px, py), radius)

    for _ in range(rng.randint(8, 16)):
        angle = rng.random() * 6.283185307179586
        reach_x = max(8, w // 2 + rng.randint(-8, 12))
        reach_y = max(8, h // 2 + rng.randint(-8, 12))
        px = int(cx + reach_x * pygame.math.Vector2(1, 0).rotate_rad(angle).x)
        py = int(cy + reach_y * pygame.math.Vector2(1, 0).rotate_rad(angle).y)
        radius = rng.randint(3, 10)
        pygame.draw.circle(surface, INVALID, (px, py), radius)


def make_world(size, seed=7):
    rng = random.Random(seed)
    surface = pygame.Surface(size)
    surface.fill(INVALID)

    for _ in range(9):
        w = rng.randint(50, 180)
        h = rng.randint(40, 150)
        x = rng.randint(0, size[0] - w - 1)
        y = rng.randint(0, size[1] - h - 1)
        draw_noisy_blob(surface, rng, (x, y, w, h))

    for _ in range(500):
        r = rng.randint(1, 4)
        x = rng.randint(r, size[0] - r - 1)
        y = rng.randint(r, size[1] - r - 1)
        pygame.draw.circle(surface, VALID, (x, y), r)

    return surface


def is_valid(surface, pos):
    x, y = int(pos[0]), int(pos[1])
    if x < 0 or y < 0 or x >= surface.get_width() or y >= surface.get_height():
        return False
    return surface.get_at((x, y))[:3] == VALID


def nearest_valid(surface, origin, max_radius=40):
    ox, oy = int(origin[0]), int(origin[1])
    if is_valid(surface, (ox, oy)):
        return ox, oy

    best = None
    best_dist2 = None
    w, h = surface.get_size()

    for r in range(1, max_radius + 1):
        left = max(0, ox - r)
        right = min(w - 1, ox + r)
        top = max(0, oy - r)
        bottom = min(h - 1, oy + r)

        for x in range(left, right + 1):
            for y in (top, bottom):
                if is_valid(surface, (x, y)):
                    dist2 = (x - ox) ** 2 + (y - oy) ** 2
                    if best_dist2 is None or dist2 < best_dist2:
                        best = (x, y)
                        best_dist2 = dist2

        for y in range(top + 1, bottom):
            for x in (left, right):
                if is_valid(surface, (x, y)):
                    dist2 = (x - ox) ** 2 + (y - oy) ** 2
                    if best_dist2 is None or dist2 < best_dist2:
                        best = (x, y)
                        best_dist2 = dist2

        if best is not None:
            return best

    return None


def clamp_pos(pos, size):
    return (
        max(0, min(size[0] - 1, pos[0])),
        max(0, min(size[1] - 1, pos[1])),
    )


def snap_nearest(mouse_pos, mouse_delta, snap_pos, valid_surface):
    _ = mouse_delta, snap_pos
    return nearest_valid(valid_surface, mouse_pos, max_radius=48)


def snap_continuity(mouse_pos, mouse_delta, snap_pos, valid_surface):
    """
    Continuity-first snapping:
      - If not currently snapped, acquire near the raw mouse.
      - If currently snapped, move the snap target by the raw cursor delta and
        search near that projected point first.
      - If that fails, try reacquiring near the raw mouse.

    Inputs:
      - mouse_pos: (x, y) from the real mouse this frame
      - mouse_delta: (dx, dy) since the previous frame
      - snap_pos: previous snapped position, or None if not snapped
      - valid_surface: binary world; VALID pixels are allowed targets

    Return:
      - new snapped position, or None if nothing valid was found
    """
    search_radius = 48

    if snap_pos is None:
        return nearest_valid(valid_surface, mouse_pos, max_radius=search_radius)

    candidate = (snap_pos[0] + mouse_delta[0], snap_pos[1] + mouse_delta[1])
    projected = nearest_valid(valid_surface, candidate, max_radius=search_radius)
    if projected is not None:
        return projected

    return nearest_valid(valid_surface, mouse_pos, max_radius=search_radius)


def snap_algorithm(mode, mouse_pos, mouse_delta, snap_pos, valid_surface):
    if mode == MODE_NEAREST:
        return snap_nearest(mouse_pos, mouse_delta, snap_pos, valid_surface)
    return snap_continuity(mouse_pos, mouse_delta, snap_pos, valid_surface)


def draw_cursor(screen, pos, snapped):
    color = CURSOR_SNAPPED if snapped else CURSOR_FREE
    x, y = int(pos[0]), int(pos[1])
    pygame.draw.circle(screen, color, (x, y), 5)
    pygame.draw.circle(screen, (0, 0, 0), (x, y), 8, 1)
    pygame.draw.line(screen, color, (x - 10, y), (x + 10, y), 1)
    pygame.draw.line(screen, color, (x, y - 10), (x, y + 10), 1)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cursor Snap Playground")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    world = make_world((WIDTH, HEIGHT))

    pygame.mouse.set_visible(False)
    prev_raw_mouse = pygame.mouse.get_pos()
    yellow_pos = prev_raw_mouse
    snap_pos = None
    snap_mode = MODE_CONTINUITY

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                world = make_world((WIDTH, HEIGHT), seed=random.randrange(1_000_000))
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_1:
                snap_mode = MODE_NEAREST
                snap_pos = None
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_2:
                snap_mode = MODE_CONTINUITY
                snap_pos = None

        raw_mouse = pygame.mouse.get_pos()
        raw_delta = (raw_mouse[0] - prev_raw_mouse[0], raw_mouse[1] - prev_raw_mouse[1])
        snapping = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        yellow_pos = clamp_pos((yellow_pos[0] + raw_delta[0], yellow_pos[1] + raw_delta[1]), (WIDTH, HEIGHT))

        if snapping:
            snap_pos = snap_algorithm(snap_mode, yellow_pos, raw_delta, snap_pos, world)
            if snap_pos is not None:
                if snap_mode == MODE_CONTINUITY:
                    yellow_pos = snap_pos
                cursor_pos = snap_pos
            else:
                cursor_pos = yellow_pos
        else:
            if snap_mode == MODE_CONTINUITY and snap_pos is not None:
                yellow_pos = snap_pos
            snap_pos = None
            cursor_pos = yellow_pos

        screen.blit(world, (0, 0))
        draw_cursor(screen, cursor_pos, snapping)

        status = "SNAP ON (hold Shift)" if snapping else "snap off (hold Shift)"
        mode_name = "1 = nearest to mouse" if snap_mode == MODE_NEAREST else "2 = continuity-first"
        info = [
            "Move mouse to drive the cursor object",
            status,
            mode_name,
            "R = regenerate blobs/noise",
            "1/2 = switch snapping algorithm",
        ]
        for i, line in enumerate(info):
            text = font.render(line, True, (240, 240, 240))
            screen.blit(text, (12, 12 + i * 18))

        pygame.display.flip()
        prev_raw_mouse = raw_mouse
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
