"""
python run.py                              # uses config.yaml
python run.py --agent autopilot --town Town05
python run.py --config my_experiment.yaml --no-display
"""
import argparse
import random

import numpy as np

from av.agents import make_agent
from av.config import load_config
from av.display import Display
from av.env import CarlaEnv


def parse_args():
    p = argparse.ArgumentParser(description="CARLA autonomous-driving framework")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--agent", help="override agent.type")
    p.add_argument("--town", help="override carla.town")
    p.add_argument("--steps", type=int, help="override run.max_steps")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.agent:
        cfg.setdefault("agent", {})["type"] = args.agent
    if args.town:
        cfg.setdefault("carla", {})["town"] = args.town
    if args.steps is not None:
        cfg.setdefault("run", {})["max_steps"] = args.steps
    if args.no_display:
        cfg.setdefault("display", {})["enabled"] = False

    if cfg.get("seed") is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    agent = make_agent(cfg.get("agent"))
    max_steps = cfg.get("run", {}).get("max_steps")

    display = None
    disp_cfg = cfg.get("display", {})
    if disp_cfg.get("enabled", True):
        name = disp_cfg.get("sensor")
        s_cfg = next((s for s in cfg.get("sensors", []) if s["name"] == name), None)
        if s_cfg is not None:
            lidar_range = float((s_cfg.get("attributes") or {}).get("range", 50))
            display = Display(name, s_cfg["type"], lidar_range)
        else:
            print(f"[run] display.sensor '{name}' not found in sensors; running without display")

    with CarlaEnv(cfg) as env:
        obs = env.reset()
        agent.setup(env)
        stats = {"collisions": 0, "lane invasions": 0}
        step = 0
        try:
            while max_steps is None or step < max_steps:
                control = agent.run_step(obs)
                obs = env.step(control)

                for name, s in env.sensors.items():
                    if s.type == "collision" and obs["sensors"][name]:
                        stats["collisions"] += len(obs["sensors"][name])
                        print(f"[run] collision with {obs['sensors'][name][0]['other_actor']}")
                    elif s.type == "lane_invasion":
                        stats["lane invasions"] += len(obs["sensors"][name])

                if display and not display.render(obs, env.ego.get_control(), agent.info, stats):
                    break
                step += 1
        except KeyboardInterrupt:
            pass
        finally:
            if display:
                display.close()
            print(f"[run] finished after {step} steps: {stats}")


if __name__ == "__main__":
    main()
