# Tennis techniques and implications for controller training

Research date: November 10, 2026. This is a practical catalogue from a short
review of USTA/LTA/ITF coaching material and a primary biomechanics study, not
an exhaustive account of tennis. It focuses on returning groundstrokes, with
other strokes and tactics included for future stages. Coaching cues are
context-dependent simplifications. **Training implications below are engineering
proposals, not rewards prescribed or validated by the coaching sources.**

## 1. What a sound groundstroke looks like

A player gets into position, prepares the body and racket early, contacts the
ball with space beside/in front of the body, controls the string-bed direction,
swings through the ball, and recovers. Beginners generally learn controlled
rallies before power or sophisticated shot selection. [LTA forehand][forehand],
[USTA backhand][backhand].

Three different geometries must not be confused:

- **Shaft direction:** the handle-to-head axis can be approximately horizontal
  at groundstroke contact. It does not need to point upward.
- **Face direction:** the string-bed plane determines whether the face is open,
  closed or approximately vertical. LTA's introductory all-purpose stroke uses
  a modestly upward-facing string bed; that is different from laying the face
  flat like a tray. [LTA shot types][shots].
- **Swing direction:** an upward-moving racket can have an approximately
  vertical/slightly closed face. The swing path and face angle are separate.
  Topspin coaching uses upward brushing; a study of 13 collegiate players linked
  greater topspin to upward racket velocity and a more closed face. Its measured
  angles are observations under controlled feeds, not universal targets.
  [USTA topspin][topspin], [Kwon et al., 2017][study].

**Implication:** the previous proposal to maximize ball/racket relative-velocity
alignment with the face normal is too restrictive as a general tennis objective.
Valid topspin includes tangential motion across the strings. Centered string-bed
contact and useful outgoing flight are better primary objectives. A normal-based
factor can be a weak anti-grazing aid, but should not prefer perfect perpendicular
incidence over every other successful impact. Likewise, do not reward a vertical
shaft or impose a single face angle across all shots.

## 2. Preparation, footwork and recovery

| Technique / coaching cue | When it helps | Possible training implication |
|---|---|---|
| Athletic ready stance: bent knees, racket in front, responsive feet | Waiting for the opponent's next shot | A soft readiness objective near the next-ball event, with a range of acceptable poses |
| Split step: a small bounce around opponent contact | Transition from recovery to reacting | Event-timed readiness, not continuous hopping |
| Long travel steps, then small setup steps | Reach the ball, then adjust spacing | Assess arrival time and final interception error separately |
| Shuffle laterally | Maintain readiness during suitable sideways movements | Evaluate recovery effectiveness; do not require one gait everywhere |
| Front-foot weight transfer as a beginner cue | Controlled, balanced introductory strokes | A possible demonstration feature, not a universal contact constraint |

These cues come from [USTA, Let's Play Tennis, p. 13][usta-feet]. The purpose of
the stance is readiness to move, not maintaining a deep static squat.

| Technique / coaching cue | When it helps | Possible training implication |
|---|---|---|
| Run, cross over or sidestep according to distance/time | Efficient court coverage | Let the motion prior provide gait variety |
| Prepare early, ideally before the bounce | Avoid rushed arm corrections | Log preparation lead time and late-contact frequency |
| Use a side-on stance when time permits; open stance under pressure | Adapt to the incoming ball | Avoid a compulsory planted front foot or fixed torso angle |
| Load the outside leg and push back after hitting | Faster recovery from a wide shot | Evaluate return toward coverage after the follow-through |

Sources: [LTA footwork][footwork], [LTA forehand][forehand].

Recovery should cover likely replies: the middle of the baseline is an
introductory reference, while the space between the opponent's possible shot
angles is more general. Restore the racket to a useful ready position before
the next shot. [LTA rallies][rallies].

**Implication for the recovery-state question:** use flexible readiness near the
next launch/opponent contact, rather than rewarding knee flexion every recovery
frame. Our recovery label also contains the end of the swing and travel back
into position. A constant squat target can conflict with those actions. In
single-ball training, a timed feed is only a surrogate for opponent contact;
full tactical recovery requires another incoming ball or an opponent model.

## 3. Contact and stroke production

| Technique / coaching cue | Practical meaning | Possible training implication |
|---|---|---|
| Contact in front and to the side | Leave room to swing instead of crowding the ball | Soft body-relative spacing, conditional on forehand/backhand |
| Beginner contact roughly knee-to-waist high | Choose an easier hitting opportunity when possible | Start feeds in a manageable band; later permit high/low contacts |
| Turn the body and prepare the racket early | Coordinate preparation, not just wrist corrections | Prefer a motion prior or demonstrations to several competing joint rewards |
| Non-dominant arm helps preparation | Supports the turn and tracking cue | Preserve coordinated source motion where available |
| Controlled swing and comfortable finish | Reliability before a maximal swing | Keep power subordinate to successful returns |

Source: [LTA forehand][forehand]. Heights and body-relative positions here are
teaching references, not hard constraints for stretched defensive shots.

| Technique / coaching cue | Practical meaning | Possible training implication |
|---|---|---|
| Control the face at impact | Direction is not determined by swing speed alone | Measure actual face orientation at impact alongside ball outcome |
| Comfortable grip and relaxed hold | Excessive tension is not the beginner goal | Do not invent a grip-force reward when the model cannot represent fingers/tension |
| One- and two-handed backhands are valid | Grip/style changes how the face is delivered | Condition demonstrations and targets on the supported stroke |
| Low-to-high topspin; high-to-low slice | Different spin intentions use different paths | Gate any swing-path prior by shot type |

Source: [USTA backhand][backhand]. Our fixed semi-western asset does not implement
human grip switching; copying grip advice literally would require model changes.

Topspin is taught to improve groundstroke margin and depth, using a brushing
action. [USTA topspin][topspin]. However, the primary study found no significant
relationship between its measured racket variables and its placement-accuracy
measures. It also did not support treating a longer straight hitting zone as
automatically better. [Kwon et al., 2017][study]. **Implication:** do not maximize
spin, upward racket velocity, or a long follow-through independently of a good
return. A follow-through is useful motion context; after separation it cannot
change that ball's trajectory.

Clean interior string-bed contact remains our engineering objective for avoiding
the simulator's rim/handle exploits. These sources do not establish a universal
sweet-spot radius or angle threshold; those must come from collider geometry,
diagnostics and validation.

## 4. Other strokes and their exceptions

| Stroke | Coaching distinction | Future training implication |
|---|---|---|
| Volley | Strike before the bounce; compact blocking action in front/to the side | Do not apply a full low-to-high groundstroke template |
| Overhead / smash | High ball taken above the head with a serve-like action | Separate contact region and preparation |
| Return of serve | Prioritize readiness and starting the rally | Evaluate reliable return coverage before winners |
| Simple introductory grip | A neutral grip can simplify starting; advanced grips vary by shot | Separate beginner teaching choices from the fixed asset's grip |

Source: [LTA shot types][shots].

| Stroke / pattern | Coaching distinction | Future training implication |
|---|---|---|
| Drop shot | Slower swing with slice/backspin into a short target | Forward-speed and deep-landing rewards must be shot-conditioned |
| Serve | Sideways setup, controlled toss, throwing action and high contact | Requires a separate task; do not reuse groundstroke contact geometry |
| Directed serves | Practise different service-box channels and spin intentions | Add serve targets only with a serve-capable model |
| Serve-return-next-ball sequence | Practise the third and fourth shots, not isolated contact | Extend evaluation to multi-shot success |

Sources: [LTA skills manual, printed pp. 3, 8–10][skills],
[LTA beginner serve][serve].

## 5. Tactics and shot selection

| Strategy | Coaching purpose | Future training implication |
|---|---|---|
| Consistency first | Sustain a rally before attacking | Over-net/in-court outcomes before tight target accuracy |
| Height and depth when defending | Buy recovery time | Permit higher defensive trajectories; no universal narrow launch band |
| Crosscourt as a common defensive option | Use the lower central net | Broad crosscourt targets with margin, not a compulsory shot |
| Attack after displacing the opponent | Exploit a favorable situation | Condition aggression on opponent position and incoming-ball difficulty |
| Hit into open space | Force movement | Needs opponent-aware targets |
| Increase pace when appropriate | Reduce opponent preparation time | Useful speed is conditional, not unlimited reward |
| Advance when attacking | Apply pressure at net | Requires net play and passing-shot responses |
| Play to strengths and opponent weaknesses | Choose effective patterns | Evaluate both sides; do not assume every opponent has a weak backhand |
| Pass, lob or target a net player's feet | Respond to an approaching opponent | Separate passing/lob/low-ball options |

Sources: [LTA tactics][tactics], [USTA backhand][backhand].

The broader LTA framework organizes tactics around consistency, space, time,
strengths and weaknesses. Its examples include creating a crosscourt opening
then changing down the line, central approaches to reduce passing angles,
aggressive second-serve returns, and drop-shot/lob combinations. Doubles adds
partner coverage and interception. [LTA skills manual, printed pp. 12–14][skills].
These require a rally/opponent environment; they cannot be learned meaningfully
from one fixed incoming flight and a universal center target alone.

## 6. How beginners practise, and the curriculum analogy

| Practice method | Coaching purpose | Controller-training analogy |
|---|---|---|
| Slower balls and appropriately smaller courts | Enable successful play early | Easier valid feeds before full distribution difficulty |
| Serve, rally and score from the first lesson | Learn the purpose of the skill while learning movement | Keep task outcomes present alongside technique shaping |
| Self-rally: tap, bounce, catch, then repeated taps | Learn height control, feet and shoulder turn | A controlled-contact subtask, not the final baseline objective |
| Short-court rally, progressively move back | Control before power | Increase required distance/difficulty gradually |
| Targets above the net | Build clearance margin | Reward a useful clearance range with landing feedback |
| Minimum rally length before competing | Build consistency | Multi-shot success and readiness for the next ball |
| Rally-length scoring and rotating game formats | Practise consistency under play conditions | Add varied game-like evaluation after basic return skill |

Sources: [ITF Play and Stay][itf], [USTA beginner self-rally][self-rally],
[LTA rallies][rallies], [LTA consistency drills][drills].

For this project, easier feeds should be regenerated/validated under the active
physics and mixed with harder cases to check transfer. A physically slower
training ball would change the dynamics; it is an optional separate experiment,
not a reason to silently modify the current tennis-ball model. Readiness
thresholds should use held-out return quality and coverage, not epoch alone.

## 7. Revised recommendation for the current stage 2 reward

The research supports the **separate clean-contact reward**, with a change to its
orientation factor:

1. **Make finite string-bed location the core quality term.** Estimate contact
   geometry during the first impact interval using the actual racket pose,
   disk radius/thickness, ball radius and contact margins. Prefer interior face
   contact smoothly; head-center proximity alone cannot distinguish the edge.
2. **Do not maximize normal incidence by default.** First log
   `abs(dot(n, v_rel)) / max(norm(v_rel), eps)` as a diagnostic. If needed, test a
   broad anti-grazing factor that saturates over acceptable incidence, with
   room for tangential brushing. There is no source-supported numeric cutoff.
   Choose one only after examining valid simulated hits across shot types.
3. **Keep successful flight as the main task.** Preserve signed forward feedback;
   add measured net/court outcomes progressively. Do not gate negative outcome
   penalties by contact quality. A broad, shot-dependent face-orientation prior
   is a later ablation if location/outcomes fail to correct tray-like impacts.
4. **Use positioning and readiness selectively.** Consider pre-contact spacing
   and event-timed recovery after basic impact quality works. Avoid a collection
   of simultaneous joint-angle rewards, an always-squat rule, or a universally
   vertical shaft rule.
5. **Test one change at a time.** Begin with a location-based clean-contact bonus,
   preserving a base contact signal. Separately test the forward/vertical balance
   and easier-feed curriculum. Compare clean-hit fraction, forward/contact,
   net/shot, in/shot and eventual multi-shot survival on matched feeds/seeds.

This revises the incidence emphasis in the earlier local review
(`tmp/contact_reward_review/review.md`); it does not change the archived analysis.
That analysis found increasing contact rate without comparable useful-return
progress in 2_4. Its solid-cylinder racket and shared handle body do not expose
exact string/edge contact labels. A geometric quality score remains a proxy,
and the rigid racket's spin response should be validated before rewarding a
specific topspin target. No reward implementation or training configuration was
changed as part of this research.

## Sources

All links were consulted on the research date. Coaching pages supply practical
cues; the study supplies limited experimental evidence, not a universal ideal.

- [LTA: How to hit a tennis forehand][forehand]
- [LTA: Types of shot][shots]
- [USTA: Learning the basics — backhand][backhand]
- [USTA: How to create topspin][topspin]
- [Kwon et al. (2017): Influence of Tennis Racquet Kinematics on Ball Topspin Angular Velocity and Accuracy during the Forehand Groundstroke][study]
- [USTA: Let's Play Tennis, second edition, p. 13][usta-feet]
- [LTA: Improve your tennis footwork][footwork]
- [LTA: How to play better tennis rallies][rallies]
- [LTA: Tennis Skills, Techniques and Tactics][skills]
- [LTA: How to hit a serve for beginners][serve]
- [LTA: Tactics][tactics]
- [ITF: Tennis Play and Stay][itf]
- [USTA: Control your forehand — beginner self-rally][self-rally]
- [LTA: Tennis drills to help build consistency][drills]

[forehand]: https://www.lta.org.uk/play/tennis-tips-and-techniques/how-to-play-a-tennis-forehand/
[shots]: https://www4.lta.org.uk/play-compete/getting-started/tennis-coaching-tips/types-of-shot/
[backhand]: https://www.usta.com/en/home/improve/tips-and-instruction/national/learning-the-basics--backhand.html
[topspin]: https://www.usta.com/en/home/improve/tips-and-instruction/national/improve-your-tennis-game--a-quick-tip-for-creating-topspin.html
[study]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5721180/
[usta-feet]: https://s3.amazonaws.com/ustaassets/assets/801/15/lets_play_tennis%21__2nd_edition_proof.pdf
[footwork]: https://www.lta.org.uk/play/tennis-tips-and-techniques/improve-your-tennis-footwork-on-court-with-these-simple-tips/
[rallies]: https://www.lta.org.uk/play/tennis-tips-and-techniques/how-to-play-better-tennis-rallies/
[skills]: https://www.lta.org.uk/4ab8ae/siteassets/roles/further-education/units-12-and-07-tennis-skills-techniques-and-tactics.pdf
[serve]: https://www.lta.org.uk/advantage-home/content/how-to-hit-a-serve-for-beginners/
[tactics]: https://www4.lta.org.uk/play-compete/getting-started/tennis-coaching-tips/tactics/
[itf]: https://www.world.tennis/en/growing-the-game/itf-tennis-play-and-stay/
[self-rally]: https://www.usta.com/en/home/improve/tips-and-instruction/national/instant-rally-forehand.html
[drills]: https://www.lta.org.uk/play/tennis-tips-and-techniques/tennis-drills-to-help-build-consistency-in-your-game/
