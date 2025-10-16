# ==============================================================================
# Melon Game with Advanced Reinforcement Learning Agent
#
# Author: Your AI Assistant
# AI Model: Dueling Double DQN with Prioritized Experience Replay (D3QN + PER)
# Key Features:
# - Hybrid Input (CNN for vision, Dense for metadata)
# - Advanced Reward Shaping
# - Play Mode use Monte Carlo tree search (PUCT)
# ==============================================================================

import pygame
import Box2D
from Box2D.b2 import world, polygonShape, circleShape, staticBody, dynamicBody, contactListener
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Input
from collections import deque, namedtuple
import math
import os
import time
import copy

# import signal
# import sys
# import shutil
# import pickle

# --- グローバル定数と設定 ---
# 基本設定
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 700
PPM = 20.0  # Pixels Per Meter
FPS = 60
TIME_STEP = 1.0 / FPS

# 色
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
WALL_COLOR = (200, 200, 220)
GAMEOVER_LINE_COLOR = (255, 0, 0)

# ゲームルール
GAMEOVER_LINE_Y = 100
ARM_POSITION_Y = 120
DROP_COOLDOWN_FRAMES = 15
WATERMELON_BONUS_SCORE = 200

# フルーツ定義
BASE_RADIUS = 8
FRUITS_DATA = [
    {'level': 0, 'name': 'cherry',     'radius': BASE_RADIUS * 1.5, 'color': (220, 20, 60),  'score': 1},
    {'level': 1, 'name': 'strawberry', 'radius': BASE_RADIUS * 2.0, 'color': (255, 99, 71),  'score': 3},
    {'level': 2, 'name': 'grape',      'radius': BASE_RADIUS * 2.5, 'color': (160, 32, 240), 'score': 6},
    {'level': 3, 'name': 'hassaku',    'radius': BASE_RADIUS * 3.0, 'color': (255, 165, 0),  'score': 10},
    {'level': 4, 'name': 'orange',     'radius': BASE_RADIUS * 3.8, 'color': (255, 140, 0),  'score': 15},
    {'level': 5, 'name': 'apple',      'radius': BASE_RADIUS * 4.5, 'color': (255, 0, 0),    'score': 21},
    {'level': 6, 'name': 'pear',       'radius': BASE_RADIUS * 5.0, 'color': (218, 247, 166),'score': 28},
    {'level': 7, 'name': 'peach',      'radius': BASE_RADIUS * 5.8, 'color': (255, 182, 193),'score': 36},
    {'level': 8, 'name': 'pineapple',  'radius': BASE_RADIUS * 6.5, 'color': (255, 255, 0),  'score': 45},
    {'level': 9, 'name': 'melon',      'radius': BASE_RADIUS * 7.5, 'color': (144, 238, 144),'score': 55},
    {'level': 10,'name': 'watermelon', 'radius': BASE_RADIUS * 8.5, 'color': (34, 139, 34),  'score': 66},
]
MAX_FRUIT_LEVEL = len(FRUITS_DATA) - 1

# AI & 強化学習向け設定
ACTION_SPACE_SIZE = 10
# 状態空間の定義
GRID_WIDTH, GRID_HEIGHT = 20, 35
GRID_CHANNELS = 2  # 0: レベル, 1: 速度(大きさ)
VECTOR_STATE_SIZE = 3 # 0: アームX座標, 1: 次フルーツレベル, 2: 次フルーツ半径

# ==============================================================================
# --- 物理エンジン用 衝突検知クラス ---
# ==============================================================================
class MyContactListener(contactListener):
    def __init__(self, env):
        super(MyContactListener, self).__init__()
        self.env = env
    
    def BeginContact(self, contact):
        bodyA, bodyB = contact.fixtureA.body, contact.fixtureB.body
        if bodyA.userData and bodyB.userData:
            levelA, levelB = bodyA.userData['level'], bodyB.userData['level']
            if levelA == levelB:
                # 合体処理の重複を避ける
                bodies_to_merge = [item['bodyA'] for item in self.env.merges_to_process] + \
                                  [item['bodyB'] for item in self.env.merges_to_process]
                if bodyA not in bodies_to_merge and bodyB not in bodies_to_merge:
                    avg_pos = (bodyA.position + bodyB.position) / 2
                    merge_info = {'bodyA': bodyA, 'bodyB': bodyB, 'pos': avg_pos, 'level': levelA}
                    self.env.merges_to_process.append(merge_info)

# ==============================================================================
# --- ゲーム環境クラス (SuikaEnv) ---
# ==============================================================================
class MelonEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Melon Game - CNN+RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        self.action_space_n = ACTION_SPACE_SIZE
        self.grid_shape = (GRID_HEIGHT, GRID_WIDTH, GRID_CHANNELS)
        self.vector_shape = (VECTOR_STATE_SIZE,)
        self.reset()

    def reset(self):
        self.world = world(gravity=(0, -30), doSleep=True)
        self.contact_listener = MyContactListener(self)
        self.world.contactListener = self.contact_listener
        self._create_walls()
        
        self.arm_x = SCREEN_WIDTH / 2
        self.score = 0
        self.game_over = False
        self.merges_to_process = []
        
        # 「直前に補充したフルーツ」のレベルを記録する変数を初期化
        # -1は「まだ何もない」ことを示す
        self.last_spawned_fruit_level = -1

        self._prepare_next_fruit()
        return self._get_state()

    def step(self, action, is_play_mode=False):
        # --- ---
        # 学習高速化のための設定（描画ありでも有効）
        if is_play_mode:
            # プレイモードの時は、人間が見やすいようにほぼ等速に
            PHYSICS_ACCELERATION = 2
        else:
            # 学習モードの時は、設定に応じて高速化する
            PHYSICS_ACCELERATION = 24  # この数値は好みに応じて調整

        # 1. アクション実行
        self.arm_x = (SCREEN_WIDTH / self.action_space_n) * (action + 0.5)
        self._drop_fruit()
        
        # 2. 物理演算の安定を待つ
        total_velocity = float('inf')
        max_wait_frames = 180
        frames_waited = 0
        while total_velocity > 0.1 and frames_waited < max_wait_frames:
            
            # --- ここから改造 ---
            # 物理計算を複数回実行
            for _ in range(PHYSICS_ACCELERATION):
                # 物理シミュレーションを1ステップ進める
                self.world.Step(TIME_STEP, 10, 10)
                # 合体処理は物理計算のたびに行う必要がある
                self._process_merges_and_removals()
            # --- ここまで改造 ---

            # 全体の速度を計算
            total_velocity = sum(body.linearVelocity.length for body in self.world.bodies if body.type == Box2D.b2_dynamicBody)
            frames_waited += 1

            # --- ここから改造 ---
            # 描画は物理計算の高速ループの外で、1回だけ行う
            if self.render_mode:
                # 描画呼び出しのたびにイベント処理も行われるため、ウィンドウが固まらない
                if not self.render(): return None, None, True, {}
            # --- ここまで改造 ---
            

        # 3. 状態遷移と報酬計算
        merge_reward, chain_count = self._process_merges_and_removals()
        
        self._check_game_over()
        
        # 報酬設計
        reward = 0
        reward += merge_reward * 1.5 # 合体によるスコア
        reward += (chain_count ** 2) * 2.0 # 連鎖ボーナス
        
        if self.game_over:
            reward -= 20.0 # ゲームオーバーペナルティ
        else:
            # 高さペナルティ（指数関数的に増加）
            max_y = 0
            for body in self.world.bodies:
                if body.userData:
                    max_y = max(max_y, body.position.y * PPM)
            if max_y > SCREEN_HEIGHT / 2:
                penalty_ratio = (max_y - SCREEN_HEIGHT / 2) / (SCREEN_HEIGHT / 2)
                reward -= 0.1 * (penalty_ratio ** 2)

        next_state = self._get_state()
        done = self.game_over
        return next_state, reward, done, {}

    def _get_state(self):
        # 状態1: 盤面グリッド (CNN用)
        grid_state = np.zeros(self.grid_shape, dtype=np.float32)
        cell_w, cell_h = SCREEN_WIDTH / GRID_WIDTH, SCREEN_HEIGHT / GRID_HEIGHT
        
        for body in self.world.bodies:
            if body.userData:
                x, y = body.position.x * PPM, body.position.y * PPM
                r = body.userData['radius']
                gx_min, gx_max = int((x - r) / cell_w), int((x + r) / cell_w)
                gy_min, gy_max = int((y - r) / cell_h), int((y + r) / cell_h)
                
                norm_level = (body.userData['level'] + 1) / (MAX_FRUIT_LEVEL + 1)
                norm_vel = np.clip(body.linearVelocity.length / 10.0, 0, 1)

                for gx in range(max(0, gx_min), min(GRID_WIDTH, gx_max + 1)):
                    for gy in range(max(0, gy_min), min(GRID_HEIGHT, gy_max + 1)):
                        gy_idx = GRID_HEIGHT - 1 - gy
                        grid_state[gy_idx, gx, 0] = max(grid_state[gy_idx, gx, 0], norm_level)
                        grid_state[gy_idx, gx, 1] = max(grid_state[gy_idx, gx, 1], norm_vel)
        
        # 状態2: メタデータベクトル (Dense用)
        norm_arm_x = self.arm_x / SCREEN_WIDTH
        norm_next_level = self.next_fruit_level / MAX_FRUIT_LEVEL
        norm_next_radius = FRUITS_DATA[self.next_fruit_level]['radius'] / FRUITS_DATA[-1]['radius']
        vector_state = np.array([norm_arm_x, norm_next_level, norm_next_radius], dtype=np.float32)

        return grid_state, vector_state

    def _create_fruit(self, level, position, apply_impulse=False):
        data = FRUITS_DATA[level].copy()
        rad_m = data['radius'] / PPM
        
        # ゲームルール変更: 小さいフルーツほど重い
        density = max(0.5, 3.0 - level * 0.25)

        body = self.world.CreateDynamicBody(
            position=position,
            fixtures=Box2D.b2FixtureDef(
                shape=circleShape(radius=rad_m),
                density=density,
                friction=0.4,
                restitution=0.2
            )
        )
        body.userData = data

        # ゲームルール変更: 合体時の反発
        if apply_impulse:
            angle = random.uniform(0, 2 * math.pi)
            force = 0.1 * body.mass
            body.ApplyLinearImpulse(impulse=(force * math.cos(angle), force * math.sin(angle)), point=body.worldCenter, wake=True)
        return body

    def _drop_fruit(self):
        pos_m = (self.arm_x / PPM, (SCREEN_HEIGHT - ARM_POSITION_Y) / PPM)
        self._create_fruit(self.next_fruit_level, pos_m)
        self._prepare_next_fruit()

    def _prepare_next_fruit(self):
        # --- ゲームバランス調整 ---
        
        # 盤面上部(上1/3)にリンゴ(レベル5)が存在するかチェック
        apple_level = 5
        orange_level = 4

        for body in self.world.bodies:
            if body.userData and body.userData['level'] == apple_level:
                # リンゴの位置をチェック
                top_y = (body.position.y * PPM) + body.userData['radius']
                if SCREEN_HEIGHT - top_y < SCREEN_HEIGHT / 3:
                    # 条件成立,高確率でオレンジを補充し、このメソッドを終了する
                    if random.random() < 0.75: # 75%の確率でオレンジが出現
                        self.next_fruit_level = orange_level
                        # last_spawned_fruit_level は更新しないでおくことで、連続出現を狙いやすくする
                        return # 処理終了
        
        
        # 2種類の出現確率リストを定義
        fruit_levels_to_spawn = [0, 1, 2, 3, 4]
        
        # 通常時の出現確率 (チェリーは出にくい)
        normal_spawn_weights = [0.10, 0.25, 0.30, 0.25, 0.10]
        
        # 直前がチェリーだった場合の出現確率 (チェリーが非常に出やすい)
        cherry_followup_weights = [0.60, 0.15, 0.10, 0.10, 0.05]

        # 状況に応じて、使用する確率リストを決定
        if self.last_spawned_fruit_level == 0:
            # 直前がチェリー(レベル0)だったので、チェリーが出やすい確率リストを使用
            weights_to_use = cherry_followup_weights
        else:
            # それ以外の場合は、通常の確率リストを使用
            weights_to_use = normal_spawn_weights

        # 重み付きランダム選択で、次のフルーツを決定
        self.next_fruit_level = random.choices(fruit_levels_to_spawn, weights=weights_to_use, k=1)[0]
        
        # 最後に、「直前のフルーツ」の記録を今決定したフルーツで更新する
        self.last_spawned_fruit_level = self.next_fruit_level

    def _process_merges_and_removals(self):
        reward, chain_count = 0.0, 0
        processed_in_this_step = set()

        merges_to_process_now = self.merges_to_process[:]
        self.merges_to_process.clear()

        for info in merges_to_process_now:
            bodyA, bodyB = info['bodyA'], info['bodyB']
            
            if bodyA in processed_in_this_step or bodyB in processed_in_this_step:
                continue
                
            if bodyA in self.world.bodies and bodyB in self.world.bodies:
                processed_in_this_step.add(bodyA)
                processed_in_this_step.add(bodyB)
                
                self.world.DestroyBody(bodyA)
                self.world.DestroyBody(bodyB)
                
                original_level = info['level'] # 合体前のフルーツのレベルを保存
                new_level = original_level + 1

                if new_level <= MAX_FRUIT_LEVEL:
                    self._create_fruit(new_level, info['pos'], apply_impulse=True)
                    score_gain = FRUITS_DATA[new_level]['score']
                    self.score += score_gain
                    reward += score_gain
                    chain_count += 1

                    # 追加ボーナス
                    # もし、合体前のフルーツがチェリー(レベル0)だったら
                    if original_level == 0:
                        cherry_clear_bonus = 5.0 # 追加で5点のボーナス報酬
                        reward += cherry_clear_bonus

                elif new_level > MAX_FRUIT_LEVEL:
                    score_gain = WATERMELON_BONUS_SCORE
                    self.score += score_gain
                    reward += score_gain
        return reward, chain_count
    
    def _check_game_over(self):
        if self.game_over: return
        for body in self.world.bodies:
            if body.userData:
                top_y = (body.position.y * PPM) + body.userData['radius']
                if SCREEN_HEIGHT - top_y < GAMEOVER_LINE_Y:
                    if body.linearVelocity.length < 0.1: # 動きが止まっていること
                        self.game_over = True
                        return

    def _create_walls(self):
        w, h, t = SCREEN_WIDTH/PPM, SCREEN_HEIGHT/PPM, 10/PPM
        self.world.CreateStaticBody(position=(w/2, 0), shapes=polygonShape(box=(w/2, t)))
        self.world.CreateStaticBody(position=(0, h/2), shapes=polygonShape(box=(t, h/2)))
        self.world.CreateStaticBody(position=(w, h/2), shapes=polygonShape(box=(t, h/2)))

    def render(self):
        if not self.render_mode: return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
        
        self.screen.fill(BLACK)
        pygame.draw.line(self.screen, GAMEOVER_LINE_COLOR, (0, GAMEOVER_LINE_Y), (SCREEN_WIDTH, GAMEOVER_LINE_Y), 2)

        for body in self.world.bodies:
            if body.userData and 'radius' in body.userData:
                x_pixel = int(body.position.x * PPM)
                y_pixel = int(SCREEN_HEIGHT - (body.position.y * PPM))
                pygame.draw.circle(self.screen, body.userData['color'], (x_pixel, y_pixel), int(body.userData['radius']))
        
        next_fruit_data = FRUITS_DATA[self.next_fruit_level]
        pygame.draw.circle(self.screen, next_fruit_data['color'], (int(self.arm_x), ARM_POSITION_Y), int(next_fruit_data['radius']))
        pygame.draw.line(self.screen, WHITE, (int(self.arm_x), 0), (int(self.arm_x), ARM_POSITION_Y), 1)
        
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(FPS)
        return True

    def close(self):
        if self.render_mode:
            pygame.quit()

    def clone(self):
        """
        物理世界を完全に再構築することで、MCTS内部工程シミュレーション用の
        独立した環境インスタンスを生成する。
        """
        # 1. 描画機能なしで、新しい空のインスタンスを作成
        cloned_env = MelonEnv(render_mode=False)
        
        # 2. 元の世界と同じ設定で、新しい物理世界を生成
        new_world = world(gravity=self.world.gravity)

        # 3. 元の世界のすべての物体(Body)をループで処理し、新しい世界に再สร้างする
        for old_body in self.world.bodies:
            # 物体の位置、角度、速度などの物理状態を完全にコピー
            new_body = new_world.CreateBody(
                type=old_body.type,
                position=old_body.position,
                angle=old_body.angle,
                linearVelocity=old_body.linearVelocity,
                angularVelocity=old_body.angularVelocity,
                fixedRotation=old_body.fixedRotation,
                userData=copy.deepcopy(old_body.userData) # userDataは辞書なのでdeepcopy可能
            )
            
            # 物体の形状や材質(Fixture)もすべてコピー
            for old_fixture in old_body.fixtures:
                old_shape = old_fixture.shape
                
                # 形状の種類に応じて、新しい形状を作成
                if isinstance(old_shape, circleShape):
                    new_shape = circleShape(radius=old_shape.radius)
                elif isinstance(old_shape, polygonShape):
                    new_shape = polygonShape(vertices=old_shape.vertices)
                # 他の形状が必要な場合はここに追加
                
                new_body.CreateFixture(
                    shape=new_shape,
                    density=old_fixture.density,
                    friction=old_fixture.friction,
                    restitution=old_fixture.restitution
                )

        # 4. 新しい世界をクローン環境に設定
        cloned_env.world = new_world
        cloned_env.contact_listener = MyContactListener(cloned_env)
        cloned_env.world.contactListener = cloned_env.contact_listener
        
        # 5. シンプルなゲーム状態変数をコピー
        cloned_env.arm_x = self.arm_x
        cloned_env.next_fruit_level = self.next_fruit_level
        cloned_env.score = self.score
        cloned_env.game_over = self.game_over
        cloned_env.last_spawned_fruit_level = self.last_spawned_fruit_level
        
        return cloned_env
    
    def destroy(self):
        """
        この環境が保持するBox2Dの世界と、その中の全オブジェクトを効果的に破棄する。
        MCTS思考工程と実行中のメモリリークを防ぐ。
        """
        if self.world:
            # イテレーション中にリストを変更すると危険なため、ボディのリストのコピーを作成する
            for body in list(self.world.bodies):
                self.world.DestroyBody(body)
        
        # オブジェクトへの参照を断ち切り、ガベージコレクタを助ける
        self.world = None
        self.contact_listener = None

    def get_board_danger_level(self):
        """
        盤面の危険度を0.0～1.0で評価する。
        「詰まり」と「不安定さ」を重視する。

        Returns:
            float: 0.0(安全) ～ 1.0(非常に危険)
        """
        if self.game_over:
            return 1.0

        all_fruits = [body for body in self.world.bodies if body.userData]
        if not all_fruits:
            return 0.0

        # --- 評価要素の計算 ---
        
        # 1. 最高到達高さ
        max_height = 0.0
        for fruit in all_fruits:
            top_y = (fruit.position.y * PPM) + fruit.userData['radius']
            max_height = max(max_height, top_y)
        
        # 2. 空間の断絶度 (Clogging Score)
        clogging_score = 0.0
        large_fruits = sorted([f for f in all_fruits if f.userData['level'] >= 4], key=lambda f: f.position.y)
        small_fruits = [f for f in all_fruits if f.userData['level'] < 4]

        # 大きなフルーツそれぞれの下に、どれだけ小さなフルーツが閉じ込められているか
        for large_fruit in large_fruits:
            lx, ly, lr = large_fruit.position.x, large_fruit.position.y, large_fruit.userData['radius'] / PPM
            for small_fruit in small_fruits:
                sx, sy = small_fruit.position.x, small_fruit.position.y
                # 小さいフルーツが、大きいフルーツの真下近くにあるか
                if sy < ly and abs(sx - lx) < lr * 1.5:
                    # 閉じ込められている度合い（高さの差が小さいほど高スコア）
                    clogging_score += (1.0 - (ly - sy) / (SCREEN_HEIGHT / PPM)) * 0.1

        # 3. 盤面の不安定さ (Velocity Score)
        total_velocity = sum(fruit.linearVelocity.length for fruit in all_fruits)
        instability_score = min(1.0, total_velocity / 50.0) # 50.0は調整可能な係数


        # --- 各要素を危険度に変換 ---

        # 危険度1: 高さによる評価 (最重要, 0.0 ～ 0.7)
        height_danger = 0.0
        # GAMEOVER_LINE_Yは画面上部からの距離なので、物理座標に変換
        danger_line_phys_y = (SCREEN_HEIGHT - GAMEOVER_LINE_Y) / PPM
        max_height_phys_y = max_height / PPM
        
        # 盤面の半分を超えたあたりから危険度を指数関数的に増加させる
        if max_height_phys_y > (SCREEN_HEIGHT / PPM) / 2:
            height_ratio = (max_height_phys_y - (SCREEN_HEIGHT / PPM) / 2) / ((SCREEN_HEIGHT / PPM) / 2)
            height_danger = 0.7 * (height_ratio ** 2)

        # 危険度2: 詰まりによる評価 (重要, 0.0 ～ 0.3)
        clogging_danger = min(0.3, clogging_score)

        # 危険度3: 不安定さによる評価 (補足的, 0.0 ～ 0.1)
        instability_danger = min(0.1, instability_score * 0.2)
        
        # 合計危険度を計算
        total_danger = height_danger + clogging_danger + instability_danger
        
        return min(1.0, total_danger)

# ==============================================================================
# --- AIエージェントクラス (D3QN + PER) ---
# ==============================================================================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos = 0

    def add(self, experience):
        max_prio = max(self.priorities) if self.buffer else 1.0
        self.buffer.append(experience)
        self.priorities.append(max_prio)

    def sample(self, batch_size):
        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + 1e-5

    def __len__(self):
        return len(self.buffer)

class D3QNPERAgent:
    def __init__(self, grid_shape, vector_shape, action_size, model_path="d3qn_per_model.h5"):
        self.grid_shape = grid_shape
        self.vector_shape = vector_shape
        self.action_size = action_size
        self.model_path = model_path
        
        self.memory = PrioritizedReplayBuffer(50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.00005
        self.update_target_freq = 1000 # steps
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.load_model()
        self.update_target_model()

    def _build_model(self):
        # Hybrid Input: CNN for grid, Dense for vector
        input_grid = Input(shape=self.grid_shape, name='grid_input')
        input_vector = Input(shape=self.vector_shape, name='vector_input')

        # CNN Path
        conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_grid)
        conv2 = layers.Conv2D(64, (3, 3), activation='relu')(conv1)
        conv3 = layers.Conv2D(64, (3, 3), activation='relu')(conv2)
        flatten = layers.Flatten()(conv3)
        cnn_features = layers.Dense(128, activation='relu')(flatten)

        # Concatenate features
        concat = layers.Concatenate()([cnn_features, input_vector])
        
        # Dueling Architecture
        dense1 = layers.Dense(256, activation='relu')(concat)
        
        # Value stream
        value_stream = layers.Dense(1, activation='linear')(dense1)
        
        # Advantage stream
        advantage_stream = layers.Dense(self.action_size, activation='linear')(dense1)
        
        # Combine streams
        q_values = value_stream + (advantage_stream - tf.reduce_mean(advantage_stream, axis=1, keepdims=True))
        
        model = models.Model(inputs=[input_grid, input_vector], outputs=q_values)
        model.compile(loss='huber', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("--- Target model updated ---")

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        grid_state, vector_state = state
        grid_state = np.expand_dims(grid_state, axis=0)
        vector_state = np.expand_dims(vector_state, axis=0)
        act_values = self.model.predict([grid_state, vector_state], verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size * 10:
            return 0.0
        
        minibatch, batch_indices, weights = self.memory.sample(batch_size)
        
        grid_states = np.array([e[0][0] for e in minibatch])
        vector_states = np.array([e[0][1] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_grid_states = np.array([e[3][0] for e in minibatch])
        next_vector_states = np.array([e[3][1] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])

        # Double DQN:
        # 1. Get the best action from the main model
        next_actions = np.argmax(self.model.predict([next_grid_states, next_vector_states], verbose=0), axis=1)
        # 2. Get the Q-value for that action from the target model
        next_q_values = self.target_model.predict([next_grid_states, next_vector_states], verbose=0)
        target_q_for_next_action = next_q_values[np.arange(batch_size), next_actions]

        targets = rewards + self.gamma * target_q_for_next_action * (1 - dones)
        
        # Update Q-values
        current_q_values = self.model.predict([grid_states, vector_states], verbose=0)
        td_errors = np.abs(targets - current_q_values[np.arange(batch_size), actions])
        self.memory.update_priorities(batch_indices, td_errors)

        current_q_values[np.arange(batch_size), actions] = targets
        
        history = self.model.fit(
            [grid_states, vector_states], current_q_values, 
            batch_size=batch_size, epochs=1, verbose=0,
            sample_weight=weights
        )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model.load_weights(self.model_path)
    
    def save_model(self):
        """
        モデルの重みを安全に保存（バックアップ付き）
        """
        import shutil
        
        backup_path = self.model_path.replace('.h5', '_backup.h5')
        
        # 既存のモデルをバックアップ
        if os.path.exists(self.model_path):
            shutil.copy2(self.model_path, backup_path)
        
        # 一時ファイルに保存してから置き換え
        try:
            temp_path = self.model_path + '.tmp'
            self.model.save_weights(temp_path)
            os.replace(temp_path, self.model_path)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Model save failed: {e}")
            # バックアップから復元
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, self.model_path)
                print("Restored model from backup")

    def save_checkpoint(self, episode, total_steps):
        """
        学習状態の完全なチェックポイントを保存
        
        Args:
            episode: 現在のエピソード番号
            total_steps: 累積ステップ数
        """
        import pickle
        import shutil
        
        checkpoint = {
            'episode': episode,
            'total_steps': total_steps,
            'epsilon': self.epsilon,
            'beta': self.memory.beta,
            'model_weights': self.model.get_weights(),
            'target_model_weights': self.target_model.get_weights(),
        }
        
        checkpoint_path = self.model_path.replace('.h5', '_checkpoint.pkl')
        backup_checkpoint_path = self.model_path.replace('.h5', '_checkpoint_backup.pkl')
        backup_model_path = self.model_path.replace('.h5', '_backup.h5')
        
        # 既存のファイルをバックアップ
        if os.path.exists(checkpoint_path):
            shutil.copy2(checkpoint_path, backup_checkpoint_path)
        if os.path.exists(self.model_path):
            shutil.copy2(self.model_path, backup_model_path)
        
        # 新しいチェックポイントを保存
        try:
            # 一時ファイルに保存してから、正常に完了したら本ファイルに置き換え
            temp_checkpoint_path = checkpoint_path + '.tmp'
            with open(temp_checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # 保存成功を確認してから本ファイルに置き換え
            os.replace(temp_checkpoint_path, checkpoint_path)
            
            print(f"Checkpoint saved: Episode {episode}, Epsilon {self.epsilon:.3f}")
        except Exception as e:
            print(f"Checkpoint save failed: {e}")
            # バックアップから復元
            if os.path.exists(backup_checkpoint_path):
                shutil.copy2(backup_checkpoint_path, checkpoint_path)
                print("Restored from backup")
    
    def load_checkpoint(self):
        """
        チェックポイントから学習状態を復元
        
        Returns:
            tuple: (開始エピソード番号, 累積ステップ数) or (0, 0)
        """
        checkpoint_path = self.model_path.replace('.h5', '_checkpoint.pkl')
        if os.path.exists(checkpoint_path):
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.epsilon = checkpoint['epsilon']
            self.memory.beta = checkpoint['beta']
            self.model.set_weights(checkpoint['model_weights'])
            self.target_model.set_weights(checkpoint['target_model_weights'])
            
            print(f"Checkpoint loaded: Episode {checkpoint['episode']}, "
                  f"Epsilon {self.epsilon:.3f}, Steps {checkpoint['total_steps']}")
            return checkpoint['episode'], checkpoint['total_steps']
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0, 0
    
    def get_policy_probs(self, state, temperature=0.5):
        """
        現在の状態から、各行動の選択確率（方策）を計算して返す。
        Q値をSoftmax関数で確率に変換する、方策ネットワークとしての機能。

        Args:
            state: 現在の環境の状態 (grid_state, vector_state)
            temperature (float): 確率分布の鋭さを調整する温度パラメータ。
                                 低いほど最善手に集中し、高いほど多様な手を考慮する。
                                 MCTSのガイドには低めの値(例: 0.5)が効果的な場合がある。
        
        Returns:
            np.ndarray: 各行動の選択確率（合計が1.0になる）
        """
        # 1. 現在のモデルでQ値を予測
        grid_state, vector_state = state
        grid_state = np.expand_dims(grid_state, axis=0)
        vector_state = np.expand_dims(vector_state, axis=0)
        q_values = self.model.predict([grid_state, vector_state], verbose=0)[0]

        # 2. Q値をSoftmax関数で確率に変換
        # temperatureで確率分布の鋭さを調整
        q_values_temp = q_values / temperature
        
        # 数値計算の安定化のため、最大値を引く（オーバーフロー防止）
        exp_q = np.exp(q_values_temp - np.max(q_values_temp))
        
        # 確率を計算
        probabilities = exp_q / np.sum(exp_q)
        
        return probabilities



# ==============================================================================
# --- モンテカルロ木探索Agentクラス (プレイモードだけで使用される) ---
# ==============================================================================
class MCTSNode:
    """MCTSのツリーを構成するノード"""
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.q_value = 0.0
        self.prior_p = prior_p # 方策ネットワークから受け取った事前確率 P(s,a)

    def is_fully_expanded(self, action_size):
        return len(self.children) == action_size

    def select_child(self, c_puct=1.0):
        """PUCTアルゴリズムに基づき、次に探索すべき子ノードを選択する"""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            # PUCTスコア = Q値(これまでの平均リターン) + U値(探索ボーナス)
            q_val = child.q_value / (child.visit_count + 1e-8)
            u_val = c_puct * child.prior_p * np.sqrt(self.visit_count) / (1 + child.visit_count)
            score = q_val + u_val
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def expand(self, action_probs):
        """ノードを展開する。方策ネットワークの確率分布に基づき子ノードを生成"""
        for action, prob in enumerate(action_probs):
            if action not in self.children and prob > 0:
                self.children[action] = MCTSNode(parent=self, prior_p=prob)

    def update(self, leaf_value):
        """葉ノードの評価値を親ノードに伝播させる (バックアップ)"""
        self.visit_count += 1
        # Q値は訪問回数で割った平均値として更新
        self.q_value += (leaf_value - self.q_value) / self.visit_count

class MCTSAgent:
    """MCTSとDQNを組み合わせたプレイモード専用エージェント"""
    def __init__(self, env, dqn_agent, num_simulations=50, c_puct=1.5):
        self.env = env
        self.dqn_agent = dqn_agent
        self.num_simulations = num_simulations
        self.c_puct = c_puct # 探索の広さを決めるハイパーパラメータ

    def act(self):
        """MCTSシミュレーションを実行し、最善の行動を返す"""
        root = MCTSNode()
        
        # MCTSのメインループ
        for i in range(self.num_simulations):
            
            # 10回に1回、Pygameのイベントキューを処理してフリーズを防ぐ
            if i % 10 == 0:
                pygame.event.pump()
            
            sim_env = None  # finallyブロックで参照
            try:
            
                # cloneメソッドを呼び出す
                sim_env = self.env.clone() 

                node = root

                # 仮想環境では、描画を完全に無効化
            
                # 1. 選択 (Selection)
                while node.is_fully_expanded(self.env.action_space_n) and node.children:
                    action, node = node.select_child(self.c_puct)
                    # シミュレーション環境も同じ行動で進める
                    sim_env.step(action, is_play_mode=False)

                # 2. 展開 (Expansion)
                # 葉ノードに到達したら、DQNに盤面を評価させる
                current_state = sim_env._get_state()
                action_probs = self.dqn_agent.get_policy_probs(current_state)
            
                # ゲームが終了していなければノードを展開
                if not sim_env.game_over:
                    node.expand(action_probs)

                # 3. 評価 (Evaluation) / ロールアウトの代わり
                # DQNのQ値予測を、その盤面の価値(leaf_value)として利用する
                # ここでは単純に最大のQ値をその盤面の価値とする
                grid, vec = current_state
                q_values = self.dqn_agent.model.predict([np.expand_dims(grid, 0), np.expand_dims(vec, 0)], verbose=0)[0]
                leaf_value = np.max(q_values)
            
                # 4. バックアップ (Backup)
                while node is not None:
                   node.update(leaf_value)
                   node = node.parent

            finally:
                # 必ず仮想世界を解体する
                if sim_env:
                    sim_env.destroy()
        
        # シミュレーション完了後、最も訪問回数が多い行動を選択する
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_action




# ==============================================================================
# --- メイン実行ブロック ---
# ==============================================================================
if __name__ == '__main__':
    # シグナルハンドラの設定（ファイルの先頭のimport文の後に追加）
    import signal
    import sys
    
    # グローバル変数で中断フラグを管理
    graceful_exit = False
    
    def signal_handler(_, __):
        """Ctrl+C や SIGTERM を受け取ったときの処理"""
        global graceful_exit
        print("\n\n 中断シグナルを受信しました。安全に終了処理を開始します...")
        print("（現在のエピソードが完了するまでお待ちください）")
        graceful_exit = True
    
    # シグナルハンドラを登録
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill コマンド
    
    # --- 既存のモード選択 ---
    mode = ""
    while mode not in ['train', 'play']:
        mode = input("Select mode ('train' or 'play'): ").lower()

    if mode == 'train':
        EPISODES = 1100
        BATCH_SIZE = 64
        RENDER_DURING_TRAINING = True

        env = MelonEnv(render_mode=RENDER_DURING_TRAINING)
        agent = D3QNPERAgent(env.grid_shape, env.vector_shape, env.action_space_n)
        
        start_episode, total_steps = agent.load_checkpoint()
        running = True
        
        # 最後に正常保存したエピソード番号を記録
        last_saved_episode = start_episode
        
        print(f"\n 学習開始: Episode {start_episode + 1} → {EPISODES}")
        print("（Ctrl+C で安全に中断できます）\n")
        
        try:  # try-except-finally ブロックで囲む
            for e in range(start_episode, EPISODES):
                # 中断フラグをチェック
                if graceful_exit:
                    print(f"\n Episode {e} で中断します...")
                    break
                
                if not running: break
                
                state = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    # エピソード途中でも中断フラグをチェック
                    if graceful_exit:
                        print(f"\n Episode {e+1} の途中で中断リクエストを受信...")
                        done = True  # エピソードを強制終了
                        break
                    
                    if not running: break

                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    if next_state is None:
                        running = False
                        continue

                    episode_reward += reward
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_steps += 1
                    
                    if RENDER_DURING_TRAINING:
                        if not env.render():
                            running = False

                    if total_steps % agent.update_target_freq == 0:
                        agent.update_target_model()

                # 中断フラグが立っている場合、保存せずにループを抜ける
                if graceful_exit:
                    break

                loss = agent.replay(BATCH_SIZE)
                
                print(f"Episode: {e+1}/{EPISODES}, Score: {env.score}, TotalReward: {episode_reward:.2f}, "
                      f"Epsilon: {agent.epsilon:.2f}, Loss: {loss:.4f}, Steps: {total_steps}")
                
                # 10エピソードごとに保存（エピソード完了後のみ）
                if (e + 1) % 10 == 0:
                    print(f" チェックポイント保存中 (Episode {e+1})...")
                    agent.save_model()
                    agent.save_checkpoint(e + 1, total_steps)
                    last_saved_episode = e + 1
                    print(f" 保存完了 (最終安全ポイント: Episode {last_saved_episode})")
        
        except KeyboardInterrupt:
            # 万が一シグナルハンドラで捕捉できなかった場合
            print("\n\n キーボード割り込みを検出しました")
            graceful_exit = True
        
        except Exception as ex:
            # 予期しないエラーをキャッチ
            print(f"\n\n エラーが発生しました: {ex}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 必ず実行される終了処理
            print("\n" + "="*60)
            print("終了処理を実行中...")
            print("="*60)
            
            if graceful_exit:
                print(f"最終安全ポイント: Episode {last_saved_episode}")
                print(f"実行中のエピソード: Episode {e+1}")
                
                # 中断時は最後の安全ポイントの状態を確認
                if last_saved_episode < e + 1:
                    print(f"Episode {last_saved_episode + 1} ～ {e+1} は保存されていません")
                    print(f"再開時は Episode {last_saved_episode + 1} から開始されます")
            else:
                # 正常終了の場合のみ最終保存
                print(" 全エピソード完了。最終保存を実行します...")
                agent.save_model()
                agent.save_checkpoint(EPISODES, total_steps)
                print(f" 最終保存完了 (Episode {EPISODES})")
            
            # 環境のクリーンアップ
            print(" 環境をクリーンアップ中...")
            env.close()
            print(" 終了")
            print("="*60 + "\n")

    elif mode == 'play':
        print("--- プレイモードを開始します (適応的MCTS搭載) ---")

        # ----------------------------------------------------
        # --- MCTSの思考回数をここで設定 ---
        # 危険度に応じて2段階で使い分け
        MCTS_SIMULATIONS_NORMAL = 7   # 通常時（危険度 < 0.4）
        MCTS_SIMULATIONS_DANGER = 50  # ピンチ時（危険度 >= 0.4）
        DANGER_THRESHOLD = 0.4         # MCTSを本格起動する危険度の閾値
        # ----------------------------------------------------

        env = MelonEnv(render_mode=True)
        
        print("直感担当のDQNエージェントをロード中...")
        dqn_agent = D3QNPERAgent(env.grid_shape, env.vector_shape, env.action_space_n)
        dqn_agent.epsilon = 0.0
        
        # MCTSエージェントは最初は作らない（必要に応じて生成）
        mcts_agent = None
        
        state = env.reset()
        done = False
        running = True
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            if not running:
                break

            # 盤面の危険度を評価
            danger_level = env.get_board_danger_level()
            print(f"盤面危険度: {danger_level:.2f}", end=" ")
            
            # 危険度に応じて行動選択方法を切り替え
            if danger_level < DANGER_THRESHOLD:
                # 安全な状態: DQNの直感だけで高速プレイ
                print("→ DQN直感モード")
                action = dqn_agent.act(state)
            else:
                # 危険な状態: MCTSで慎重に思考
                # MCTSエージェントがまだない場合は作成
                if mcts_agent is None:
                    print(f"\n 危険！MCTS起動 (シミュレーション: {MCTS_SIMULATIONS_DANGER}回)")
                    mcts_agent = MCTSAgent(env, dqn_agent, num_simulations=MCTS_SIMULATIONS_DANGER)
                else:
                    # 既にある場合はシミュレーション回数を更新
                    mcts_agent.num_simulations = MCTS_SIMULATIONS_DANGER
                
                print("→ MCTS慎重モード", end=" ")
                start_time = time.time()
                action = mcts_agent.act()
                end_time = time.time()
                print(f"(思考時間: {end_time - start_time:.2f}秒)")
            
            # 行動実行
            next_state, _, done, _ = env.step(action, is_play_mode=True)
            state = next_state

            if next_state is None:
                running = False

        print(f"\nプレイ終了! 最終スコア: {env.score}")
        time.sleep(3)
        env.close()