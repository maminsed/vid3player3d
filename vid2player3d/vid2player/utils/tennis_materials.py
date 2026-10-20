"""Initialize collision materials before PhysX actors are instantiated.

IsaacGym's GPU material cache can retain the asset defaults when materials are
changed only on actors. Keep the generator and the live ball on the same path.
This module deliberately has no IsaacGym dependency so it can be tested on CPU.
"""


def configure_ball_asset(gym, asset, restitution=0.9, friction=0.2, contact_offset=None):
    props = gym.get_asset_rigid_shape_properties(asset)
    for prop in props:
        prop.restitution = restitution
        prop.friction = friction
        prop.compliance = 0.5
        if contact_offset is not None:
            prop.contact_offset = contact_offset
            prop.rest_offset = 0.
    gym.set_asset_rigid_shape_properties(asset, props)


def configure_racket_asset(gym, asset, racket_body_id, restitution, friction, contact_offset=None):
    # Use body-to-shape ranges rather than assuming one collider per body.
    ranges = gym.get_asset_rigid_body_shape_indices(asset)
    racket = ranges[racket_body_id]
    if racket.count != 2:
        raise ValueError('Expected racket handle and head collision shapes')
    props = gym.get_asset_rigid_shape_properties(asset)
    head = props[racket.start + 1]
    head.restitution = restitution
    head.friction = friction
    head.compliance = 0.5
    if contact_offset is not None:
        head.contact_offset = contact_offset
        head.rest_offset = 0.
    gym.set_asset_rigid_shape_properties(asset, props)
