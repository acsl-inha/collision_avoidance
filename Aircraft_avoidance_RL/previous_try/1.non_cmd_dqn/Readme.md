# non_cmd_dqn
본 코드는 pytorch tutorial의 DQN 학습 코드를 기반으로 작성되었다. 환경은 GYM의 Custom env툴을 사용하여 제작하였다. pytorch tutorial의 자세한 내용은 아래 링크를 참조하라.
https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html

먼저 가장 간단한 실험을 위해 고도 변화 명령을 주는 term에만 관여하여 reward를 지정하였다.
reward는 에피소드가 종료될때 고도 변화 명령값의 절대값을 누적으로 더한 값에 -1을 곱하여 받게 된다.
에피소드 종료 조건은 
1. 목표 항공기와 본체의 거리가 200미터 미만일때
2. 목표 항공기가 본체의 시야에서 벗어났을때
3. 목표 항공기와 본체의 거리가 5000미터 이상일때
4. 시뮬레이션 지속시간이 30초 이상일때

로 총 4가지이다.


# 구현 결과

## Rewards after 10000 episodes 
<img src="../res_img/non_cmd_reward.JPG" width="40%">

## Results after 10000 episodes (hdot_cmd, h, r, elev, azim)
<img src="../res_img/non_cmd_res.JPG" width="40%">

## 3D plot
<img src="../res_img/non_cmd_3d.JPG" width="40%">

구현 결과를 보면, Reward의 수렴이 잘 이루어지지 않았다. 따라서 기존에 구현되어 있는 cartpole 환경처럼 매 스텝마다 reward를 제공해보려 한다. 

[2.non_cmd_dqn_step_reward](../2.non_cmd_dqn_step_reward)
