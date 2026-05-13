import gymnasium as gym

env = gym.make(
    "browsergym/openended",
    task_kwargs={"start_url": "https://www.google.com/"},
    wait_for_user_message=True,
    headless=False,
)

obs, info = env.reset()
done = False
while not done:
    action = 'send_msg_to_user("What would you like me to do?")'
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
