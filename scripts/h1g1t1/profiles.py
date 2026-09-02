#!/usr/bin/env python3
"""Validated H1+G1+T1 experiment profiles.

The current URMA2 environment interprets ``environment.nr_envs`` as the total
across all train robots.  All profiles therefore choose totals divisible by
three and minibatches that preserve equal per-family contribution.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    total_envs: int
    rollout_steps: int
    minibatch_size: int
    ppo_epochs: int
    total_timesteps: int
    save_frequency: int
    morphology_target: float
    morphology_ramp_steps: int
    eval_envs: int
    eval_steps: int

    @property
    def robot_count(self) -> int:
        return 3

    @property
    def envs_per_family(self) -> int:
        return self.total_envs // self.robot_count

    @property
    def batch_size(self) -> int:
        return self.total_envs * self.rollout_steps

    @property
    def minibatches_per_epoch(self) -> int:
        return self.batch_size // self.minibatch_size

    @property
    def samples_per_family_per_minibatch(self) -> int:
        return self.minibatch_size // self.robot_count

    @property
    def timesteps_per_family(self) -> int:
        return self.total_timesteps // self.robot_count

    @property
    def updates(self) -> int:
        return self.total_timesteps // self.batch_size

    def validate(self) -> None:
        errors: list[str] = []
        if self.total_envs <= 0 or self.rollout_steps <= 0:
            errors.append("environment and rollout sizes must be positive")
        if self.total_envs % self.robot_count:
            errors.append("total_envs must be divisible by three robot families")
        if self.minibatch_size % self.robot_count:
            errors.append("minibatch_size must be divisible by three")
        if self.batch_size % self.minibatch_size:
            errors.append("batch_size must be divisible by minibatch_size")
        if self.total_timesteps % self.batch_size:
            errors.append("total_timesteps must be a multiple of batch_size")
        if self.save_frequency % self.batch_size:
            errors.append("save_frequency must be a multiple of batch_size")
        if self.total_timesteps % self.save_frequency:
            errors.append("total_timesteps must be a multiple of save_frequency")
        if not 0.0 <= self.morphology_target <= 1.0:
            errors.append("morphology_target must be in [0, 1]")
        if not 0 < self.morphology_ramp_steps <= self.total_timesteps:
            errors.append("morphology_ramp_steps must be in (0, total_timesteps]")
        if self.eval_envs % self.robot_count:
            errors.append("eval_envs must be divisible by three")
        if errors:
            raise ValueError(f"invalid profile {self.name}: " + "; ".join(errors))


_PROFILES = {
    # Compilation/integration check. It is not a learning-quality experiment.
    "smoke": Profile(
        name="smoke",
        total_envs=96,
        rollout_steps=64,
        minibatch_size=3072,
        ppo_epochs=5,
        total_timesteps=614_400,
        save_frequency=307_200,
        morphology_target=0.10,
        morphology_ramp_steps=307_200,
        eval_envs=24,
        eval_steps=200,
    ),
    # 800 PPO rollouts; 9.8304M samples per family.
    "probe": Profile(
        name="probe",
        total_envs=576,
        rollout_steps=64,
        minibatch_size=6144,
        ppo_epochs=5,
        total_timesteps=29_491_200,
        save_frequency=2_949_120,
        morphology_target=0.30,
        morphology_ramp_steps=17_694_720,
        eval_envs=96,
        eval_steps=1000,
    ),
    # 4000 PPO rollouts; 49.152M samples per family.
    "full": Profile(
        name="full",
        total_envs=576,
        rollout_steps=64,
        minibatch_size=6144,
        ppo_epochs=5,
        total_timesteps=147_456_000,
        save_frequency=14_745_600,
        morphology_target=0.30,
        morphology_ramp_steps=88_473_600,
        eval_envs=96,
        eval_steps=1000,
    ),
}


def get_profile(name: str) -> Profile:
    try:
        profile = _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; choose {sorted(_PROFILES)}") from exc
    profile.validate()
    return profile


def as_dict(profile: Profile) -> dict[str, object]:
    data = dataclasses.asdict(profile)
    data.update(
        robot_count=profile.robot_count,
        envs_per_family=profile.envs_per_family,
        batch_size=profile.batch_size,
        minibatches_per_epoch=profile.minibatches_per_epoch,
        samples_per_family_per_minibatch=profile.samples_per_family_per_minibatch,
        timesteps_per_family=profile.timesteps_per_family,
        updates=profile.updates,
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=sorted(_PROFILES))
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    data = as_dict(get_profile(args.profile))
    if args.shell:
        for key, value in data.items():
            shell_key = key.upper()
            text = str(value).lower() if isinstance(value, bool) else str(value)
            print(f"{shell_key}={shlex.quote(text)}")
    else:
        print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
