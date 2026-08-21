# Claude Prompt for Auto-Struct Capstone

You are helping with a capstone project focused on sequential text-to-video generation.

## Project Idea

The core problem is this:

When we generate a video from a long sequential sentence or paragraph, some events get mixed up, some events are skipped, and the final video does not preserve the intended order.

The project solves this by:

1. Breaking a long instruction paragraph into simpler atomic sentences.
2. Mapping each atomic sentence to a single action.
3. Generating a separate short video clip for each action.
4. Stitching the clips together in the correct order.
5. Preserving consistency across all clips, especially:
   - same person
   - same background
   - same camera viewpoint
   - same controlled environment

The main focus for now is handwashing videos.

## Problem Statement

Develop an automated text-to-video generation system that accurately models event order, timing, and motion to produce coherent, strictly sequential multi-event videos from natural language descriptions.

## Scope

- Process natural language instructions containing multiple sequential sentences
- Enforce strict temporal alignment such that only one event is active at any time
- Generate short video clips for each event using a pretrained text-to-video model
- Limit the system to a single fixed daily activity: handwashing
- Use a controlled recording environment with a static camera and consistent viewpoint
- Use pretrained video generation models without training a new model from scratch

## Assumptions

- Only one event is active at any given time
- The system is limited to a single daily activity, mainly handwashing
- Videos are recorded using a static camera with a fixed viewpoint and background
- Input instructions are well-formed, sequential, and contain explicit ordering cues such as first, then, and finally
- The meaning of each action, such as "apply soap", is consistent across inputs and videos
- Temporal alignment relies on a dataset with sentence-level annotations
- The final output depends on reliable video concatenation and synchronization tools

## Proposed Methodology

### Step 1: Sentence-Level Decomposition

- Split the input instructional paragraph into ordered atomic sentences
- Treat each sentence as a single executable event
- Preserve ordering cues such as first, then, and finally

### Step 2: Action Mapping Using Controlled Action Space

- Map each sentence to a predefined action category
- Ensure paraphrased instructions map to the same action abstraction
- Use this to improve linguistic robustness

### Step 3: Temporal Assignment and Enforcement

- Assign a duration to each action using learned or rule-based priors
- Enforce a strict non-overlap constraint between consecutive actions
- Ensure only one action is active at any time

### Step 4: Action-Level Video Retrieval or Generation

- For each action, select or generate a corresponding atomic video clip
- Each clip should represent only one step, not the entire task
- Use a pretrained text-to-video model for rendering
- VideoCrafter or similar tools can be considered

### Step 5: Sequential Assembly at Inference Time

- Stitch action-level clips in the predicted temporal order
- Respect durations using trimming or stretching
- Produce a final video that strictly follows the instruction sequence
- Use tools such as ffmpeg or moviepy for concatenation

## Current Progress in My Folder

I have already started building this project in my current workspace.

Current files include:

- `parser.py`
- `scheduler.py`
- `run_pipeline.py`
- `test_parser.py`
- `test_scheduler.py`
- `handwash_dataset.csv`
- `README.md`

What the current code is doing:

- `parser.py` performs instruction parsing and action extraction
- `scheduler.py` assigns durations to actions using the dataset and LLM reasoning
- `run_pipeline.py` runs the parser and scheduler together
- `test_parser.py` and `test_scheduler.py` contain sample tests

The current implementation is mainly focused on handwashing instructions.

## What I Want From You

Please do the following:

1. Propose a strong architecture for this project.
2. Explain the end-to-end pipeline clearly.
3. Suggest the best approach for solving the sequential action decomposition problem.
4. Suggest how to preserve consistency across clips.
5. Suggest how to handle handwashing as a controlled domain.
6. Recommend whether the system should use:
   - a rule-based approach
   - an LLM-based approach
   - a hybrid approach
7. Give me 2 or 3 architecture options if useful, then recommend the best one.
8. Keep the solution practical for a capstone project.
9. Be specific about modules, data flow, and responsibilities of each stage.
10. Highlight any risks, limitations, and what can be improved later.

## Desired Output Format

Please respond with:

- A clean architecture diagram in text form
- A step-by-step approach
- Module breakdown
- Recommended model/tool stack
- Data flow from input text to final stitched video
- Risks and limitations
- Best final recommendation

Keep the answer focused on my handwashing video use case, but make the design general enough to extend later.
