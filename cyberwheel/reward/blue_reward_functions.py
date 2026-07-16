
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


def reward_red_delay_availability(rewarder, **kwargs):
    """reward_red_delay plus an availability cost: every benign (green) event
    blocked this step because its source or destination host is isolated
    costs ``blocked_event_penalty`` (env config key, default 1.0). This is
    what makes indiscriminate quarantining a losing strategy for blue when a
    green agent is active — the penalty recurs naturally for as long as the
    host stays isolated AND users keep trying to reach it, so an idle host
    stays cheap to quarantine while a busy server is expensive.
    """
    b, b_recurring = reward_red_delay(rewarder, **kwargs)

    green_agent_result = kwargs.get("green_agent_result", None)
    if green_agent_result is not None and green_agent_result.events_blocked:
        penalty = getattr(rewarder.args, "blocked_event_penalty", 1.0)
        b -= penalty * green_agent_result.events_blocked

    return b, b_recurring
