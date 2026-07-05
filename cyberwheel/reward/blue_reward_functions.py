
def reward_red_delay(rewarder, **kwargs):
    blue_agent_result = kwargs.get("blue_agent_result", None)
    red_agent_result = kwargs.get("red_agent_result", None)

    b = 0
    b_recurring = 0

    if red_agent_result.success and red_agent_result.target_host.decoy:
        b += 100
    if blue_agent_result.success:
        b += rewarder.blue_rewards[blue_agent_result.name][0]
        # A recurring=1 action (e.g. quarantine) emits its configured recurring
        # cost, applied every subsequent step until a recurring=-1 pops it.
        # Actions whose YAML recurring is 0 emit 0, so existing configs are
        # unchanged.
        if blue_agent_result.recurring == 1:
            b_recurring = rewarder.blue_rewards[blue_agent_result.name][1]
    if blue_agent_result.id == "decoy_limit_exceeded":
        b += -5000
    else:
        pass

    return b, b_recurring