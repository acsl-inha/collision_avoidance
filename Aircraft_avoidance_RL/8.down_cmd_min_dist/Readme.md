# down_cmd_min_dist
아래 방향으로의 학습을 위해, 고도 변화 명령에 관한 reward를 빼고, 충돌에 관련한 reward만을 남겨두었다. "5.down_cmd_dqn_long_sim"과 마찬가지로 시나리오 종료조건이 완화된 상태로 수정되어 학습되었고, reward는 충돌시 -100을 반환하고, 충돌 하지 않은 경우 매 스텝마다 1을 반환하도록 구현되었다.

# 구현 결과

## Rewards after 10000 episodes (Moving average 200)
<img src="../res_img/down_cmd_min_dist_reward.png" width="40%">

## Results after 10000 episodes (hdot_cmd, h, r, elev, azim)
<img src="../res_img/down_cmd_min_dist_res.png" width="40%">

## 3D plot
<img src="../res_img/down_cmd_min_dist_3d.png" width="40%">

## Height plot
<img src="../res_img/down_cmd_min_dist_height.png" width="40%">

결과를 보면, 의도와는 달리 수렴 결과가 위로 회피하는 결과를 보였다. 생각해보니 수렴 결과의 단일성이 없어서 그렇다는 결론이 나왔다. 위로 회피하던, 아래로 회피하던 둘 다 동일한 값을 reward를 받게 되기 때문에, 상대기가 본체보다 높은 고도에서 출발했다 하더라도, 그보다 더 높게 이동하여 회피 가능한 상황이 존재하면, 위로 회피하는 경우도 높은 reward를 받기 때문이다. 상대기가 본체보다 높은 고도에서 출발했을때, 이 요소가 수렴에 관여하려면 아무래도 종료 시점의 distance가 reward에 영향을 미쳐야 한다고 생각되어 reward를 수정하여 다시 실험을 진행하였다.

[9.min_dist_cmd_retry](../9.min_dist_cmd_retry)
