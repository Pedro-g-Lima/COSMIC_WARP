"""Draw the README's diagrams into ``docs/assets/``, light and dark SVG each.

``step``: one ``simulate()`` step and the fused integrator kernel inside it
(README, ``docs/theory/execution.md``). ``learning``: one ``DiffVecEnv``
window, forward and through ``backward()`` (README,
``docs/guides/differentiable-rl.md``). They are committed, since GitHub cannot
run the docs build, and drawn here rather than by hand so the stage lists
below are the one source: ``tests/test_diagram.py`` holds them to the code's
call order, and fails when a committed file was drawn by an older version of
this file. That check reads a hash each SVG records of this file, not the
SVG's bytes: the bytes differ across Python and library versions for the
same drawing.

    python scripts/diagrams.py   # after changing a diagram or the style

Needs matplotlib (the ``dev`` or ``docs`` group); importing this module does not.
"""
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "docs" / "assets"

# One simulate() step, in launch order (SpacecraftSimulator.simulate's
# step_body, with Integrator.step opened), and the fused kernel's phases, in
# the order every integrator kernel runs them. A "\n" is only a line break in
# the drawing; the test compares the labels with it read as a space.
STEP = ["step kernel", "multi-body\nreduction", "store\ntrajectory", "auto-reset", "trajectory\nindex"]
STEP_NOTES = {"step kernel": "integrator, fused", "multi-body\nreduction": "if num_bodies > 1",
              "store\ntrajectory": "if save_states", "auto-reset": "if auto_reset_terminated",
              "trajectory\nindex": "one thread: +1"}
KERNEL = ["episode\ncounter +1", "terminated?", "thruster\nwrench", "integration\nstages",
          "store state,\nwheel momenta", "termination\ncheck"]
KERNEL_NOTES = {"thruster\nwrench": "once per step"}
# What each integration stage evaluates: the terms summed into the force and
# the torque, and the wheels.
FORCES = ["gravity + J2", "drag", "thrust"]
TORQUES = ["thrust", "gravity gradient", "magnetic"]
# Roles, by colour: termination and resets apart from the rest.
TERMINATION = {"multi-body\nreduction", "auto-reset", "terminated?", "termination\ncheck"}
OPTIONAL = {"multi-body\nreduction", "store\ntrajectory", "auto-reset"}

# GitHub's inks, so the drawings sit on its pages; the reference categorical
# palette for the roles (docs/stylesheets/extra.css brings it to the site).
# Fills are translucent, so one file suits any background of its scheme
# (GitHub's and Material's differ).
THEMES = {
    "light": {"ink": "#1f2328", "muted": "#59636e", "line": "#8c959f", "edge": "#d0d7de", "fill": 0.12,
              "region": ("#1f2328", 0.03), "panel": ("#ffffff", 0.9),
              "user": "#eb6834", "step": "#2a78d6", "physics": "#1baf7a", "grad": "#c98500"},
    "dark": {"ink": "#e6edf3", "muted": "#9198a1", "line": "#6e7681", "edge": "#3d444d", "fill": 0.20,
             "region": ("#e6edf3", 0.04), "panel": ("#e6edf3", 0.03),
             "user": "#d95926", "step": "#3987e5", "physics": "#199e70", "grad": "#eda100"},
}
W = 900   # drawing units: 1 unit = 1/100 inch
HEAD = "-|>,head_length=0.45,head_width=0.22"
DASH = (0, (4, 3))


def _box(ax, x, y, w, h, t, role, label=None, radius=7, dashed=False, fontsize=10.5):
    from matplotlib.colors import to_rgb
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=(*to_rgb(t[role]), t["fill"]), ec=t[role], lw=1.3,
                                ls=(0, (3, 2)) if dashed else "-"))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color=t["ink"],
                linespacing=1.15)


def _noted_box(ax, x, y, w, h, t, role, label, note, dashed=False):
    """A box whose label sits above an italic note on what it is for."""
    _box(ax, x, y, w, h, t, role, dashed=dashed)
    ax.text(x + w / 2, y + (h + 18) / 2, label, ha="center", va="center", fontsize=10.5, color=t["ink"],
            linespacing=1.1)
    ax.text(x + w / 2, y + 10, note, ha="center", va="center", fontsize=8.5, color=t["muted"],
            style="italic")


def _arrow(ax, p0, p1, color, lw=1.1):
    ax.annotate("", p1, p0, arrowprops=dict(arrowstyle=HEAD, color=color, lw=lw, shrinkA=0, shrinkB=0))


def _grad_arrow(ax, p0, p1, t):
    """A dashed gradient line with one solid head at its end."""
    import numpy as np

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    ax.plot(*zip(p0, p1 - 8 * u), color=t["grad"], lw=1.5, ls=DASH)
    _arrow(ax, p1 - 9 * u, p1, t["grad"], lw=1.5)


def _legend(ax, t, y, entries):
    x = 12
    for role, text in entries:
        if role == "grad":
            ax.plot([x, x + 14], [y, y], color=t["grad"], lw=1.6, ls=DASH)
        elif role == "optional":
            _box(ax, x + 1, y - 6, 12, 12, t, "step", radius=3, dashed=True)
        else:
            _box(ax, x + 1, y - 6, 12, 12, t, role, radius=3)
        x = _right(ax, ax.text(x + 21, y, text, va="center", fontsize=9.5, color=t["muted"])) + 22


def _right(ax, text):
    """Where a text ends, in drawing units."""
    return ax.transData.inverted().transform(text.get_window_extent())[1][0]


def _opens(ax, t, top, bottom):
    """Dashed lines from a box's lower corners to the panel it opens into."""
    (a0, a1, ya), (b0, b1, yb) = top, bottom
    for a, b in ((a0, b0), (a1, b1)):
        ax.plot([a, b], [ya, yb], color=t["physics"], lw=1, ls=(0, (3, 3)))


def _panel(ax, t, x, y, w, h):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9",
                                fc=t["panel"], ec=t["physics"], lw=1.1, ls=(0, (3, 3))))


def _step(ax, t, H):
    from matplotlib.patches import Circle, FancyBboxPatch

    _legend(ax, t, H - 16, (("step", "the step's kernels, in launch order"),
                            ("physics", "inside the fused kernel, one thread per entity"),
                            ("user", "termination and reset"), ("optional", "optional")))
    role = lambda label, default: "user" if label in TERMINATION else default   # noqa: E731

    # One simulate() step: a tinted region, named at its foot.
    rx, ry, rw, rh = 10, 10, W - 20, H - 46
    ax.add_patch(FancyBboxPatch((rx, ry), rw, rh, boxstyle="round,pad=0,rounding_size=12",
                                fc=t["region"], ec="none"))
    name = ax.text(rx + 16, ry + 26, "one simulate() step", va="bottom", fontsize=9.5, color=t["ink"],
                   weight="bold")
    ax.text(_right(ax, name) + 8, ry + 26,
            "on CUDA, recorded once per call as a CUDA graph and replayed num_steps times",
            va="bottom", fontsize=9.5, color=t["muted"])
    ax.text(rx + 16, ry + 8, "sim.step() launches the first two kernels, eagerly; VecEnv.step() runs the "
            "step kernel inside its own graph", va="bottom", fontsize=9.5,
            color=t["muted"])

    # The step's kernels, each with what it is for.
    bw, bh, gap = 158, 58, 18
    pw = len(STEP) * bw + (len(STEP) - 1) * gap
    x0 = rx + (rw - pw) / 2
    y = ry + rh - 54 - bh
    xs = [x0 + i * (bw + gap) for i in range(len(STEP))]
    for x, label in zip(xs, STEP):
        _noted_box(ax, x, y, bw, bh, t, role(label, "step"), label, STEP_NOTES[label],
                   dashed=label in OPTIONAL)
    for a, b in zip(xs, xs[1:]):
        _arrow(ax, (a + bw, y + bh / 2), (b, y + bh / 2), t["line"])

    # The loop back to the next step, above the row.
    yl = y + bh + 22
    xa, xb = xs[-1] + bw / 2, xs[0] + bw / 2
    ax.plot([xa, xa, xb], [y + bh, yl, yl], color=t["line"], lw=1.1)
    _arrow(ax, (xb, yl), (xb, y + bh), t["line"])
    ax.text((xa + xb) / 2, yl + 5, "next step: the graph replayed", ha="center", va="bottom", fontsize=9.5,
            color=t["muted"])

    # The step kernel opened up: what each thread does.
    px, ptop, pbottom = x0, y - 30, ry + 66
    _panel(ax, t, px, pbottom, pw, ptop - pbottom)
    _opens(ax, t, (xs[0], xs[0] + bw, y), (px + 0.5, px + pw - 0.5, ptop))
    title = ax.text(px + 16, ptop - 10, "the step kernel", va="top", fontsize=10, color=t["ink"],
                    weight="bold")
    ax.text(_right(ax, title) + 10, ptop - 10, "one launch, one thread per entity", va="top", fontsize=9.5,
            color=t["muted"])

    kw, kh, kg = 118, 54, 20
    end_room = 34   # where a thread ends, after the row
    kx0 = px + (pw - end_room - (len(KERNEL) * kw + (len(KERNEL) - 1) * kg)) / 2
    ky = ptop - 66 - kh
    kxs = [kx0 + i * (kw + kg) for i in range(len(KERNEL))]
    for x, label in zip(kxs, KERNEL):
        if label in KERNEL_NOTES:
            _noted_box(ax, x, ky, kw, kh, t, "physics", label, KERNEL_NOTES[label])
        else:
            _box(ax, x, ky, kw, kh, t, role(label, "physics"), label, fontsize=9.5)
    for a, b in zip(kxs, kxs[1:]):
        _arrow(ax, (a + kw, ky + kh / 2), (b, ky + kh / 2), t["line"])
    end = (kxs[-1] + kw + 20, ky + kh / 2)
    _arrow(ax, (kxs[-1] + kw, end[1]), (end[0] - 4, end[1]), t["line"])
    ax.add_patch(Circle(end, 4, fc=t["ink"], ec="none"))

    # A terminated entity skips the rest: its state stays frozen until a reset.
    term = kxs[KERNEL.index("terminated?")] + kw / 2
    yb = ky + kh + 18
    ax.plot([term, term, end[0], end[0]], [ky + kh, yb, yb, end[1] + 12], color=t["user"], lw=1.2)
    _arrow(ax, (end[0], end[1] + 13), (end[0], end[1] + 4.5), t["user"], lw=1.2)
    ax.text(term + 10, yb + 4, "yes: skip to the end, the state frozen until a reset",
            va="bottom", fontsize=9, color=t["muted"])

    # The integration stages opened up: what each stage evaluates.
    sx = kxs[KERNEL.index("integration\nstages")]
    cw, ch, cg = 236, 54, 16
    sw = 3 * cw + 2 * cg + 32
    spx = min(max(sx + kw / 2 - sw / 2, px + 12), px + pw - 12 - sw)
    sptop = ky - 30
    spy = pbottom + 14
    _panel(ax, t, spx, spy, sw, sptop - spy)
    _opens(ax, t, (sx, sx + kw, ky), (spx + 0.5, spx + sw - 0.5, sptop))
    ax.text(spx + 16, sptop - 10,
            "each stage evaluates: Euler once, symplectic at both kicks, RK4 four times",
            va="top", fontsize=9.5, color=t["muted"])
    cards = (("forces, inertial", " · ".join(FORCES)), ("torques, body", " · ".join(TORQUES)),
             ("wheel momenta", "closed form at the stage time"))
    for k, (head, detail) in enumerate(cards):
        cx = spx + 16 + k * (cw + cg)
        _box(ax, cx, spy + 12, cw, ch, t, "physics")
        ax.text(cx + cw / 2, spy + 12 + ch / 2 + 9, head, ha="center", va="center", fontsize=10,
                color=t["ink"])
        ax.text(cx + cw / 2, spy + 12 + ch / 2 - 10, detail, ha="center", va="center", fontsize=8.5,
                color=t["muted"])


def _learning(ax, t, H):
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon

    _legend(ax, t, H - 16, (("user", "your code: policy, task, loss"),
                            ("step", "DiffSimStepper.step(): Warp kernels"),
                            ("grad", "loss.backward(): gradients")))

    # The chain of one window: three steps shown, the middle ones elided.
    yc, r, sw, sh, ag = 250, 18, 84, 38, 13
    labels = [r"$\boldsymbol{x}_0$", r"$\boldsymbol{x}_1$", r"$\boldsymbol{x}_2$", r"$\boldsymbol{x}_{H-1}$",
              r"$\boldsymbol{x}_H$"]
    nodes, steps, x = [], [], 104
    for k in range(5):
        nodes.append(x)
        x += r + ag
        if k == 2:
            x += 40 - ag   # the elided steps
        elif k < 4:
            steps.append(x)
            x += sw + ag
        x += r
    start_x = nodes[0] - r - 22

    # One environment's card, with copies behind it for the others. The copies
    # are outlines clipped to where they show, so the file stays transparent.
    bx0, bx1, by0, by1 = 26, nodes[-1] + r + 22, 128, 302
    card = dict(boxstyle="round,pad=0,rounding_size=12", lw=1)
    shows = Polygon([(0, by1), (bx1, by1), (bx1, 0), (W, 0), (W, H), (0, H)], transform=ax.transData)
    for d in (16, 8):
        copy = ax.add_patch(FancyBboxPatch((bx0 + d, by0 + d), bx1 - bx0, by1 - by0, fc="none", ec=t["edge"],
                                           **card))
        copy.set_clip_path(shows)
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0, fc=t["region"], ec=t["edge"], **card))
    name = ax.text(bx0 + 4, by0 - 12, "env.rollout(policy): one window of H steps", va="top", fontsize=9.5,
                   color=t["ink"], weight="bold")
    ax.text(_right(ax, name) + 8, by0 - 12, "of one environment; all N step together in the same launches",
            va="top", fontsize=9.5, color=t["muted"])

    for nx, label in zip(nodes, labels):
        ax.add_patch(Circle((nx, yc), r, fc="none", ec=t["ink"], lw=1.2))
        ax.text(nx, yc, label, ha="center", va="center", fontsize=10 if "H-1" not in label else 8.5,
                color=t["ink"])
    for k, sx in enumerate(steps):
        before = nodes[k if k < 2 else 3]
        _box(ax, sx, yc - sh / 2, sw, sh, t, "step", "step")
        _arrow(ax, (before + r, yc), (sx, yc), t["line"])
        _arrow(ax, (sx + sw, yc), (nodes[k + 1 if k < 2 else 4] - r, yc), t["line"])
    # The elided steps between x_2 and x_{H-1}.
    gx0, gx1 = nodes[2] + r, nodes[3] - r
    ax.text((gx0 + gx1) / 2, yc, "⋯", ha="center", va="center", fontsize=13, color=t["muted"])
    # The window starts from the carried state: a bar, where backward stops.
    ax.plot([start_x, start_x], [yc - 22, yc + 22], color=t["step"], lw=1.8)
    ax.text(start_x - 4, yc - 27, "carried in,\ndetached", ha="left", va="top", fontsize=8.5,
            color=t["muted"], linespacing=1.1)

    # The policy, shared: observations up, actions down, parameter gradients up.
    py, ph = 362, 32
    px0, px1 = nodes[0] - r, steps[-1] + sw
    _box(ax, px0, py, px1 - px0, ph, t, "user", radius=8)
    ax.text((px0 + px1) / 2, py + ph / 2,
            r"policy $\pi_{\boldsymbol{\theta}}$: observation $\rightarrow$ action, one $\boldsymbol{\theta}$ "
            r"for every step and environment",
            ha="center", va="center", fontsize=10, color=t["ink"])
    mid = by1 + 30   # labels above the card and its copies
    for k, sx in enumerate(steps):
        nx = nodes[k if k < 2 else 3]
        _arrow(ax, (nx, yc + r), (nx, py), t["line"])
        ax.text(nx - 4, mid + 6, r"$o_t$", ha="right", va="center", fontsize=9.5, color=t["muted"])
        xa, xg = sx + sw / 2 - 12, sx + sw / 2 + 12
        _arrow(ax, (xa, py), (xa, yc + sh / 2), t["line"])
        ax.text(xa - 4, mid, r"$\boldsymbol{a}_t$", ha="right", va="center", fontsize=9.5, color=t["muted"])
        _grad_arrow(ax, (xg, yc + sh / 2), (xg, py), t)
        ax.text(xg + 5, mid, r"$\partial L/\partial\boldsymbol{\theta}$", ha="left", va="center",
                fontsize=9.5, color=t["grad"])

    # The window's end: the final state carries over, detached.
    _arrow(ax, (nodes[-1] + r, yc), (bx1 + 40, yc), t["line"])
    ax.text(bx1 + 46, yc + 10, "env.end_window():\n" + r"$\boldsymbol{x}_H$ carried to the next" + "\nwindow, detached",
            va="center", fontsize=9, color=t["muted"], linespacing=1.25)

    # Rewards into the loss.
    yr = 150
    lbx, lbw, lby0, lby1 = bx1 + 40, 178, 134, 228
    for sx in steps:
        _arrow(ax, (sx + sw / 2 - 12, yc - sh / 2), (sx + sw / 2 - 12, yr), t["line"])
        ax.text(sx + sw / 2 - 16, yr + 14, r"$r_t$", ha="right", va="center", fontsize=9.5, color=t["muted"])
    ax.plot([steps[0] + sw / 2 - 12, bx1 + 20], [yr, yr], color=t["line"], lw=1.1)
    _arrow(ax, (bx1 + 20, yr), (lbx, yr), t["line"])
    _box(ax, lbx, lby0, lbw, lby1 - lby0, t, "user", radius=8)
    ax.text(lbx + lbw / 2, (lby0 + lby1) / 2,
            "loss $L$, your trainer's\n" + r"e.g. SHAC: $-\Sigma_t\,\gamma^t r_t$" + "\n"
            + r"$-\ \gamma^H V(o_H)$, mean over" + "\nthe N environments",
            ha="center", va="center", fontsize=9.5, color=t["ink"], linespacing=1.35)

    # Backward through the whole window, back to the carried state.
    yg = 186
    _grad_arrow(ax, (lbx, yg), (start_x + 2, yg), t)
    ax.text((nodes[2] + nodes[3]) / 2, yg - 6, "backward through every step", ha="center", va="top",
            fontsize=8.5, color=t["grad"])

    for y, text in ((68, "backward() runs through the policy, the task's observation and reward, and each step's "
                         "Warp adjoint, back to the start of the window."),
                    (48, "The carried state is detached, so no gradient crosses a window boundary. Every environment "
                         "resets together at the time limit,"),
                    (30, "a multiple of H, outside autograd. Termination checks are off on this path, so episodes "
                         "end only by truncation.")):
        ax.text(bx0 + 4, y, text, va="center", fontsize=9, color=t["muted"])


DIAGRAMS = {"step": (_step, 540), "learning": (_learning, 450)}   # drawing, height


def draw(name, theme):
    """One diagram as a matplotlib figure; the caller saves and closes it."""
    import matplotlib
    matplotlib.use("agg")   # text is measured while drawing; the SVG renderer measures at its own 72 dpi
    import matplotlib.pyplot as plt

    body, height = DIAGRAMS[name]
    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "svg.fonttype": "path", "svg.hashsalt": "cosmicwarp", "savefig.transparent": True}):
        fig = plt.figure(figsize=(W / 100, height / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, height)
        ax.axis("off")
        body(ax, THEMES[theme], height)
    return fig


def svg(name, theme):
    """One diagram as SVG text."""
    import io

    import matplotlib.pyplot as plt

    fig = draw(name, theme)
    buffer = io.StringIO()
    with plt.rc_context({"svg.fonttype": "path", "svg.hashsalt": "cosmicwarp"}):
        fig.savefig(buffer, format="svg", transparent=True,
                    metadata={"Date": None, "Description": f"scripts/diagrams.py {source_hash()}"})
    plt.close(fig)
    return buffer.getvalue()


def source_hash():
    """What a committed SVG records of the code that drew it."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def path(name, theme):
    return ASSETS / f"{name}-{theme}.svg"


if __name__ == "__main__":
    for name in DIAGRAMS:
        for theme in THEMES:
            path(name, theme).write_text(svg(name, theme), encoding="utf-8")
            print(f"written {path(name, theme).relative_to(REPO)}")
