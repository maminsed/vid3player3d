# Vid3Player3d

A machine learning and reinforcement learning pipeline for training an autonomous agent to play tennis in a 3D simulated environment.

---

# Overview

## Problem

Teaching simulated agents realistic sports movements is difficult. It requires accurate motion extraction from video, reliable physics simulation, and reinforcement learning that can turn recorded human movement into controllable agent behavior.

## Goal

The goal of this project is to build a virtual tennis-playing agent that improves on NVIDIA's [vid2player3d](https://research.nvidia.com/labs/toronto-ai/vid2player3d/) project. We aim to create more natural player movement, better racket control, and stronger overall gameplay.

## Approach

This project builds on the approach introduced in NVIDIA's vid2player3d paper. We extract human motion data from real tennis broadcast footage, process and refine the motion data, then use reinforcement learning to train an agent in a physics simulation to reproduce those movements realistically.

---

# Current State

The full data processing and extraction pipeline is now implemented. The current workflow processes raw YouTube videos through the following stages:

1. **Formatting & Segmentation**: Converts raw videos into smaller clips that are easier to analyze.
2. **Feature Extraction**: Uses the `TennisProject` module to track players and the ball and export game coordinates to CSV files.
3. **3D Pose Reconstruction**: Processes clips through `GVHMR` to generate 3D human pose reconstructions.
4. **Data Filtering**: Removes low-quality or incorrect reconstructions.
5. **Coordinate Correction**: Uses `vid2player3d` to correct and project global trajectory coordinates.

Current development is focused on training an encoder-decoder model that converts the extracted motion sequences into physically realistic agent movement.

---

# Project Structure

The workspace is split into three main modules, each responsible for a different stage of the pipeline:

```bash
.
├── GVHMR/          # Generates 3D human pose reconstructions from video clips.
├── TennisProject/  # Handles player/ball tracking, court registration, and bounce detection.
└── vid2player3d/   # Corrects trajectories, runs the physics simulation, and trains the final agent.
```

---

# Installation

To set up the project, you will need to install [conda](https://docs.conda.io/en/latest/miniconda.html).
You must create three separate conda environments—one for each of the main directories—and follow their respective installation instructions:

- **GVHMR:** Follow the [GVHMR README](GVHMR/README.md) to set up its environment.
- **TennisProject:** Follow the [TennisProject README](TennisProject/README.md) to set up its environment.
- **vid2player3d:** Follow the [vid2player3d README](vid2player3d/README.md) to set up its environment.


<!-- # Results

## Metrics

| Metric | Value |
|---|---|
| <Metric> | <Value> |
| <Metric> | <Value> |

---

## Benchmarks

| Model | Score |
|---|---|
| <Baseline> | <Score> |
| <Your Model> | <Score> |

--- -->
