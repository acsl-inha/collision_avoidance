# reward_retry_min_dist
이전의 min_dist에서 사용했던 방식의 reward를 추가하여 재실험하였다.
 
 # 구현 결과
## Rewards after 10000 episodes (Moving average 200)
<img src="../res_img/down_cmd_step_reward.png" width="40%">

## Results after 10000 episodes (hdot_cmd, h, r, elev, azim)
<img src="../res_img/down_cmd_step_res.png" width="40%">

## 3D plot
<img src="../res_img/down_cmd_step_3d.png" width="40%">

## Height plot
<img src="../res_img/down_cmd_step_height.png" width="40%">
 
다시 조정이 필요해 보인다. 생각을 해보니 hcmd를 적게 사용하도록 잘 수렴시킨다면, 최소거리 회피에 대한 reward를 따로 추가하지 않아도 되어야 한다. 따라서 이 term을 제외하고, 또 hcmd를 주는 시점이 에피소드 종료 직전 즈음인지, 에피소드 시작 지점인지에 대한 부분도 단일성이 보장되지 않기 때문데, 에피소드의 time step에 대한 term을 reward에 포함시켜 적용해보려 한다.
 
 [14.reward_retry_add_time](../14.reward_retry_add_time)
