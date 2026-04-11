## Requirements Document

### Project: **Projected Expectations**

Build an interactive digital artwork called **Projected Expectations** that runs **locally on a laptop** and uses the laptop camera plus machine learning to detect a person standing in front of the camera. The system should visually separate the person’s body from the background using a **colored body mask**, then generate floating text labels around the body based on the system’s **analyzed gender presentation** of the person in frame.

The final piece should feel visually striking, playful, and expressive. Bright colors are welcome. The work should still communicate social labeling and projected expectations, but with a more vivid and stylized visual language.

---

## Concept

When a person stands in front of the camera, the program should:

* detect the person
* segment or mask their body so it is visually separated from the background
* estimate or classify the person into a gender category for the purpose of the artwork
* generate floating labels around the body based on that analyzed gender
* animate those labels in real time as the person moves

The goal is to create the feeling that bodies are being automatically read, categorized, and surrounded by expectations.

---

## Technical Direction

Preferred implementation:

* **Python**
* **OpenCV**
* **MediaPipe** for pose detection
* A **segmentation model** for separating the person from the background
* Optional lightweight gender classification model if needed
* Rendering can be done with OpenCV drawing functions, Pygame, or another local graphics approach

The project should run locally on a laptop with a webcam.

---

## Functional Requirements

### 1. Camera Input

* Access the laptop webcam
* Display the camera feed in a local application window
* Mirror the camera feed so it behaves like a mirror

### 2. Human Segmentation / Body Mask

* Detect the person in front of the camera
* Create a visible colored mask over the person’s body
* The mask should clearly separate the person from the background
* The mask color should be bright and artistic, not neutral
* The background should remain visible, but the person should stand out strongly

Examples:

* neon pink body mask
* cyan silhouette overlay
* gradient or shifting color fill if feasible

The body mask should move with the person in real time.

### 3. Pose Tracking

Track the person’s body to support label placement.

At minimum track:

* head or face position
* shoulders
* torso center
* wrists if possible

This data should be used to place and animate text labels around the masked body.

### 4. Gender Analysis

The system should analyze the person standing in front of the camera and assign them into a gendered category for the purpose of choosing labels.

For implementation purposes, this can be:

* feminine
* masculine
* and optionally an ambiguous or mixed state if confidence is low

The coding agent should use a lightweight local gender classification method if possible.

If the prediction confidence is low or unstable:

* either blend label sets
* or switch to a mixed category
* avoid rapid flickering between categories

### 5. Label Pools by Gender Category

#### Feminine coded label pool

* smile more
* pretty
* too emotional
* be modest
* likable
* soft
* quiet
* nurturing
* polite
* desirable
* graceful
* delicate

#### Masculine coded label pool

* strong
* leader
* dominant
* confident
* aggressive
* ambitious
* tough
* stoic
* powerful
* provider
* assertive
* in control

#### Mixed / unstable / ambiguous pool

* too much
* difficult
* confusing
* not enough
* perform
* explain yourself
* choose
* readable
* wrong
* undefined

The system should select labels from the chosen pool and place them around the body.

### 6. Label Behavior

Each label should:

* fade in
* float around the body
* follow body movement with slight lag
* drift slightly for an organic feel
* remain readable

Labels should appear near:

* head
* shoulders
* upper torso
* surrounding body space

### 7. Presence Behavior

When a person enters the frame:

* activate the mask
* determine a gender category
* fade in an initial set of labels
* attach them around the body

When no person is detected:

* remove or fade out the mask
* fade out all labels
* return to idle state

### 8. Movement Behavior

When the person moves:

* mask updates in real time
* labels follow the body
* some labels scatter briefly and reform

When the person remains still:

* labels hover more steadily
* the composition should still feel active and alive

### 9. Distance Behavior

Use body size or shoulder width to estimate closeness to camera.

If the person moves closer:

* increase number of labels
* make labels cluster closer to the body
* intensify the visual pressure

If the person moves farther:

* reduce label density
* spread labels outward

---

## Visual Requirements

The piece should use a **bright, playful, expressive color palette**.

Requirements:

* bright body mask colors
* colorful floating labels
* visually exciting composition
* strong contrast between person and background
* motion should feel fluid and artistic

Possible palette direction:

* hot pink
* bright cyan
* yellow
* lime green
* purple
* orange

The work can feel playful on the surface while still carrying critical meaning.

---

## Implementation Steps

### Step 1

Set up a local camera application in Python.

### Step 2

Add person detection and segmentation so the body can be separated from the background with a colored overlay or silhouette mask.

### Step 3

Add pose tracking to estimate:

* face or head position
* shoulder points
* torso center
* wrists if available

### Step 4

Add a local gender analysis step that classifies the detected person into a category used for label selection.

### Step 5

Create three label pools:

* feminine coded
* masculine coded
* mixed / ambiguous

### Step 6

Create a label system where each label has:

* text
* position
* target anchor
* opacity
* drift
* follow behavior
* fade state

### Step 7

Spawn labels around the body based on the selected gender category.

### Step 8

Update labels in real time so they:

* follow the body
* drift
* scatter during movement
* increase in density when the person moves closer

### Step 9

Render the final composition with:

* mirrored camera feed
* bright body mask
* floating colorful labels

### Step 10

Add smoothing and stability logic so segmentation, pose tracking, and gender category do not flicker too aggressively.

---

## Deliverables

Please provide:

* a working local Python project
* code organized into multiple files where reasonable
* setup instructions
* dependency list
* commented code
* editable label lists
* editable color settings

---

## Nice to Have

* animated color shifting in the body mask
* labels with different font sizes
* mixed label mode when classification confidence is low
* debug mode that shows segmentation and landmarks
* presentation mode with clean final visuals only

---

## Important Note for the Agent

The gender classification in this project is being used as an artistic mechanism to critique automatic categorization and social projection. The result does not need to claim objective truth. It should function as part of the conceptual logic of the artwork.
