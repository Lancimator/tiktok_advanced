import os
import sys
import threading
import queue
import random
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox


SIM_TYPES = [
    ("three_hit", "3-Hit Rings + Freeze (balls stack)"),
    ("duo", "Dual Ball Bouncy (balls collide)"),
    ("duo_freeze", "Dual Ball + Freeze (stacking)"),
    # Future options can be added here, e.g. ("spiral", "Spiral Reveal"), ...
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Reveal Generator")
        # Center window and raise to front so it's visible
        w, h = 640, 520
        try:
            self.update_idletasks()
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            x = max(0, (sw - w) // 2)
            y = max(0, (sh - h) // 3)
            self.geometry(f"{w}x{h}+{x}+{y}")
        finally:
            pass
        self.lift()
        try:
            self.attributes('-topmost', True)
            self.after(400, lambda: self.attributes('-topmost', False))
        except Exception:
            pass

        self.msg_q = queue.Queue()
        self.worker = None
        self.stop_flag = threading.Event()

        # Defaults
        self.videos_dir_var = tk.StringVar(value="videos")
        self.out_dir_var = tk.StringVar(value="finished")
        self.count_var = tk.StringVar(value="1")
        self.ring_lives_var = tk.StringVar(value="3")

        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm_top = ttk.Frame(self)
        frm_top.pack(fill=tk.X, **pad)

        # How many videos
        ttk.Label(frm_top, text="How many videos:" ).grid(row=0, column=0, sticky=tk.W)
        self.count_entry = ttk.Spinbox(frm_top, from_=1, to=999, textvariable=self.count_var, width=8)
        self.count_entry.grid(row=0, column=1, sticky=tk.W, padx=8)

        # Videos dir
        ttk.Label(frm_top, text="Videos folder:").grid(row=1, column=0, sticky=tk.W)
        self.videos_entry = ttk.Entry(frm_top, textvariable=self.videos_dir_var, width=40)
        self.videos_entry.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=8)

        # Output dir
        ttk.Label(frm_top, text="Output folder:").grid(row=2, column=0, sticky=tk.W)
        self.out_entry = ttk.Entry(frm_top, textvariable=self.out_dir_var, width=40)
        self.out_entry.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=8)

        # Simulation types
        frm_types = ttk.Labelframe(self, text="Simulation types (choose one or more)")
        frm_types.pack(fill=tk.X, **pad)
        self.type_vars: dict[str, tk.BooleanVar] = {}
        self.type_cbs: dict[str, ttk.Checkbutton] = {}
        for key, label in SIM_TYPES:
            var = tk.BooleanVar(value=(key == "three_hit"))
            self.type_vars[key] = var
            cb = ttk.Checkbutton(frm_types, text=label, variable=var)
            cb.pack(anchor=tk.W, padx=8, pady=2)
            self.type_cbs[key] = cb

        # Fully random option (moved to above Start button). It randomizes mode, ring lives,
        # ball life, balls count, and future simulation-defining parameters.
        self.fully_random_var = tk.BooleanVar(value=False)

        # Three-hit settings
        frm_opts = ttk.Frame(self)
        frm_opts.pack(fill=tk.X, **pad)
        self.ring_lives_label = ttk.Label(frm_opts, text="Ring lives:")
        self.ring_lives_label.pack(side=tk.LEFT)
        self.ring_lives_entry = ttk.Spinbox(frm_opts, from_=1, to=20, textvariable=self.ring_lives_var, width=6)
        self.ring_lives_entry.pack(side=tk.LEFT, padx=8)

        # Mode options
        frm_mode = ttk.Labelframe(self, text="Mode options")
        frm_mode.pack(fill=tk.X, **pad)
        # Amount of balls (for duo/duo_freeze)
        ttk.Label(frm_mode, text="Amount of balls:").grid(row=0, column=0, sticky=tk.W)
        self.balls_count_var = tk.StringVar(value="2")
        self.balls_count_entry = ttk.Spinbox(frm_mode, from_=1, to=20, textvariable=self.balls_count_var, width=6)
        self.balls_count_entry.grid(row=0, column=1, sticky=tk.W, padx=8)
        # Ball life seconds (for three_hit/duo_freeze)
        ttk.Label(frm_mode, text="Ball life (seconds):").grid(row=1, column=0, sticky=tk.W)
        self.ball_life_var = tk.StringVar(value="2.0")
        self.ball_life_entry = ttk.Spinbox(frm_mode, from_=0.5, to=30.0, increment=0.5, textvariable=self.ball_life_var, width=6)
        self.ball_life_entry.grid(row=1, column=1, sticky=tk.W, padx=8)

        # Randomization toggle just above the Start button
        frm_rand = ttk.Frame(self)
        frm_rand.pack(fill=tk.X, **pad)
        self.fully_random_cb = ttk.Checkbutton(
            frm_rand,
            text="Fully Random (random mode, ring lives, ball life, balls, etc.)",
            variable=self.fully_random_var,
        )
        self.fully_random_cb.pack(anchor=tk.W)

        # Controls
        frm_ctrl = ttk.Frame(self)
        frm_ctrl.pack(fill=tk.X, **pad)
        self.start_btn = ttk.Button(frm_ctrl, text="Start", command=self.on_start)
        self.start_btn.pack(side=tk.LEFT)
        self.quit_btn = ttk.Button(frm_ctrl, text="Quit", command=self.destroy)
        self.quit_btn.pack(side=tk.LEFT, padx=8)

        # Progress
        self.prog = ttk.Progressbar(self, mode="determinate")
        self.prog.pack(fill=tk.X, **pad)

        # Log
        frm_log = ttk.Labelframe(self, text="Log")
        frm_log.pack(fill=tk.BOTH, expand=True, **pad)
        self.log = tk.Text(frm_log, height=16, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)

        # React to Fully Random toggle by greying out other options
        try:
            self.fully_random_var.trace_add("write", lambda *args: self._update_random_state())
        except Exception:
            # Fallback for older Tk versions
            self.fully_random_var.trace("w", lambda *args: self._update_random_state())
        self._update_random_state()
        # Also watch mode checkboxes to update which options to enable
        for var in self.type_vars.values():
            try:
                var.trace_add("write", lambda *args: self._update_random_state())
            except Exception:
                var.trace("w", lambda *args: self._update_random_state())

    def _log(self, msg: str):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def set_running(self, running: bool):
        state = tk.DISABLED if running else tk.NORMAL
        self.start_btn.config(state=state)
        self.count_entry.config(state=state)
        self.videos_entry.config(state=state)
        self.out_entry.config(state=state)
        for var in self.type_vars.values():
            pass  # checkbuttons remain clickable is okay; could disable if desired

    def _update_random_state(self):
        """Enable/disable options based on Fully Random toggle."""
        fr = bool(self.fully_random_var.get())
        # Grey out type checkboxes
        for cb in self.type_cbs.values():
            try:
                if fr:
                    cb.state(["disabled"])
                else:
                    cb.state(["!disabled"])
            except Exception:
                cb.configure(state=(tk.DISABLED if fr else tk.NORMAL))
        # Grey out ring lives controls
        try:
            if fr:
                self.ring_lives_label.state(["disabled"])  # visual cue
            else:
                self.ring_lives_label.state(["!disabled"])
        except Exception:
            pass
        self.ring_lives_entry.configure(state=("disabled" if fr else "normal"))
        # Grey out/enable mode specifics based on selection
        sel = [k for k, v in self.type_vars.items() if v.get()]
        any_duo = any(k in ("duo", "duo_freeze") for k in sel)
        any_life = any(k in ("three_hit", "duo_freeze") for k in sel)
        # When fully random, disable all; else enable per rules
        self.balls_count_entry.configure(state=("disabled" if fr or not any_duo else "normal"))
        self.ball_life_entry.configure(state=("disabled" if fr or not any_life else "normal"))

    def on_start(self):
        # Validate inputs
        try:
            count = int(self.count_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid number of videos.")
            return
        if count < 1:
            messagebox.showerror("Invalid input", "Number of videos must be at least 1.")
            return

        videos_dir = self.videos_dir_var.get().strip() or "videos"
        out_dir = self.out_dir_var.get().strip() or "finished"
        selected = [k for k, v in self.type_vars.items() if v.get()]
        if not selected:
            # default to all available if none selected
            selected = [k for k, _ in SIM_TYPES]

        # parse ring lives
        try:
            ring_lives = max(1, int(self.ring_lives_var.get()))
        except ValueError:
            ring_lives = 3

        os.makedirs(out_dir, exist_ok=True)

        # Start worker thread
        self.set_running(True)
        self.prog.config(maximum=count, value=0)
        self._log(f"Starting generation: {count} video(s), types={selected}, videos_dir='{videos_dir}', out='{out_dir}'")
        self.stop_flag.clear()
        fully_random = bool(self.fully_random_var.get())
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(count, videos_dir, out_dir, selected, ring_lives, fully_random),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, count: int, videos_dir: str, out_dir: str, types: list[str], ring_lives: int, fully_random: bool):
        py = sys.executable or "python"
        ok = True
        for i in range(count):
            if self.stop_flag.is_set():
                break
            # Choose type and ring lives per video
            if fully_random:
                available = [k for k, _ in SIM_TYPES]
                t = random.choice(available)
                ring_lives_i = random.randint(1, 5)
                ball_life_i = round(random.uniform(2.0, 8.0), 2)
                # Random initial balls for duo/duo_freeze
                balls_count_i = random.randint(1, 6)
            else:
                t = random.choice(types)
                ring_lives_i = ring_lives
                ball_life_i = None
                try:
                    ball_life_i = float(self.ball_life_var.get())
                except Exception:
                    pass
                try:
                    balls_count_i = max(1, int(self.balls_count_var.get()))
                except Exception:
                    balls_count_i = 2
            label = next((lbl for k, lbl in SIM_TYPES if k == t), t)
            if fully_random:
                self.msg_q.put(("log", f"[{i+1}/{count}] Running simulation: {label} (ring lives={ring_lives_i}, ball life={ball_life_i}s)"))
            else:
                if t in {"duo", "duo_freeze"}:
                    self.msg_q.put(("log", f"[{i+1}/{count}] Running simulation: {label} (ring lives={ring_lives_i}, balls={balls_count_i}, ball life={ball_life_i if ball_life_i is not None else '-'}s)"))
                else:
                    self.msg_q.put(("log", f"[{i+1}/{count}] Running simulation: {label} (ring lives={ring_lives_i}, ball life={ball_life_i if ball_life_i is not None else '-'}s)"))

            # Build command for selected mode
            if t in {"three_hit", "duo", "duo_freeze"}:
                cmd = [
                    py, os.path.join("scripts", "sim_reveal.py"),
                    videos_dir, out_dir,
                    "--mode", t,
                    "--ring-lives", str(ring_lives_i),
                ]
                if ball_life_i is not None:
                    cmd += ["--ball-life", str(ball_life_i)]
                if t in {"duo", "duo_freeze"} and balls_count_i is not None:
                    cmd += ["--balls", str(balls_count_i)]
            else:
                # Fallback
                cmd = [py, os.path.join("scripts", "sim_reveal.py"), videos_dir, out_dir]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    self.msg_q.put(("log", f"  Error: return code {proc.returncode}\n{proc.stderr.strip()}"))
                    ok = False
                else:
                    out_line = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else "(no output)"
                    self.msg_q.put(("log", f"  Done: {out_line}"))
            except Exception as e:
                self.msg_q.put(("log", f"  Exception: {e}"))
                ok = False

            self.msg_q.put(("progress", "1"))

        self.msg_q.put(("done", "ok" if ok else "fail"))

    def _poll_queue(self):
        try:
            while True:
                what, payload = self.msg_q.get_nowait()
                if what == "log":
                    self._log(payload or "")
                elif what == "progress":
                    self.prog.step(1)
                elif what == "done":
                    self.set_running(False)
                    status = "Success" if (payload == "ok") else "Completed with errors"
                    self._log(status)
                    messagebox.showinfo("Finished", status)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
