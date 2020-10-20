# reward_retry_add_time
 회피 기동의 단일성을 위해 reward에 time step term을 추가하였다. 이를 통해 hcmd 명령을 더 일찍 주는것이 더 높은 reward를 받도록 구현하였다. 수정한 reward는 다음과 같다.
 
 - 회피 시 100의 reward 반환
 - 충돌 시 -50의 reward 반환
 - 매 스텝마다 hcmd의 절대값에 -0.05를 곱하여 반환
 
 # 구현 결과
## Rewards after 10000 episodes (Moving average 200)
<img src="../res_img/down_cmd_step_reward.png" width="40%">

## Results after 10000 episodes (hdot_cmd, h, r, elev, azim)
<img src="../res_img/down_cmd_step_res.png" width="40%">

## 3D plot
<img src="../res_img/down_cmd_step_3d.png" width="40%">

## Height plot
<img src="../res_img/down_cmd_step_height.png" width="40%">
 
 여전히 수렴이 잘 진행되지 않았다. hcmd를 주지 않았을때 회피 하는 경우(애초에 충돌조건에 해당되지 않는 에피소드)가 존재하여
 이 부분을 최대한 제거하고 거의 무조건 충돌조건이도록 환경을 수정하여 실험해 보려 한다.
 
 [15.hcmd_modulate](../15.hcmd_modulate)
