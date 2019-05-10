from __future__ import division

import multiprocessing

from os import path

import pygame

# -------------------------------  #
import cv2
import sys

import os
import math
import random

import tensorflow as tf

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

# -------------------------------  #

import glob, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# -------------------------------  #

###############################

sys.path.append("..")

q = multiprocessing.Queue()

#############################


def set_object_center():

    # classes:
    # 1.Aeroplanes     2.Bicycles   3.Birds       4.Boats           5.Bottles
    # 6.Buses          7.Cars       8.Cats        9.Chairs          10.Cows
    # 11.Dining tables 12.Dogs      13.Horses     14.Motorbikes     15.People
    # 16.Potted plants 17.Sheep     18.Sofas      19.Trains         20.TV/Monitors

    detect_class = 15

    slim = tf.contrib.slim

    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)

    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net = ssd_vgg_300.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

    # Restore SSD model.
    # ckpt_filename = 'checkpoints/ssd_300_vgg.ckpt'
    ckpt_filename = '../SSD-Tensorflow/checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'

    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)

    # Main image processing routine.
    def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                                  feed_dict={img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes

    def get_bboxes(rclasses, rbboxes):
        # get center location of object

        number_classes = rclasses.shape[0]
        object_bboxes = []
        for i in range(number_classes):
            object_bbox = dict()
            object_bbox['i'] = i
            object_bbox['class'] = rclasses[i]
            object_bbox['y_min'] = rbboxes[i, 0]
            object_bbox['x_min'] = rbboxes[i, 1]
            object_bbox['y_max'] = rbboxes[i, 2]
            object_bbox['x_max'] = rbboxes[i, 3]
            object_bboxes.append(object_bbox)
        return object_bboxes

    # load net
    net = SiamRPNvot()
    net.load_state_dict(torch.load(join(realpath(dirname(__file__)), '../DaSiamRPN-master/code/SiamRPNVOT.model')))

    net.eval()

    # open video capture
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    index = True
    while index:

        # Read first frame.
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        # Define an initial bounding box
        height = frame.shape[0]
        width = frame.shape[1]

        rclasses, rscores, rbboxes = process_image(frame)

        bboxes = get_bboxes(rclasses, rbboxes)
        for bbox in bboxes:
            if bbox['class'] == detect_class:
                print(bbox)
                ymin = int(bbox['y_min'] * height)
                xmin = int((bbox['x_min']) * width)
                ymax = int(bbox['y_max'] * height)
                xmax = int((bbox['x_max']) * width)
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                h = ymax - ymin
                w = xmax - xmin
                new_bbox = (cx, cy, w, h)
                print(new_bbox)
                index = False
                break

    # tracker init
    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    state = SiamRPN_init(frame, target_pos, target_sz, net)

    # tracking and visualization
    toc = 0
    count_number = 0

    while True:

        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        tic = cv2.getTickCount()

        # Update tracker
        state = SiamRPN_track(state, frame)  # track
        # print(state)

        toc += cv2.getTickCount() - tic

        if state:

            res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            res = [int(l) for l in res]
            cv2.rectangle(frame, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
            count_number += 1
            # set object_center
            object_center = dict()
            object_center['x'] = state['target_pos'][0]/width
            object_center['y'] = state['target_pos'][1]/height
            q.put(object_center)

            if (not state) or count_number % 40 == 3:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                            2)
                index = True
                while index:
                    ok, frame = video.read()
                    rclasses, rscores, rbboxes = process_image(frame)
                    bboxes = get_bboxes(rclasses, rbboxes)
                    for bbox in bboxes:
                        if bbox['class'] == detect_class:
                            ymin = int(bbox['y_min'] * height)
                            xmin = int(bbox['x_min'] * width)
                            ymax = int(bbox['y_max'] * height)
                            xmax = int(bbox['x_max'] * width)
                            cx = (xmin + xmax) / 2
                            cy = (ymin + ymax) / 2
                            h = ymax - ymin
                            w = xmax - xmin
                            new_bbox = (cx, cy, w, h)
                            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                            state = SiamRPN_init(frame, target_pos, target_sz, net)

                            index = 0

                            break

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    video.release()
    cv2.destroyAllWindows()


def game():

    # assets folder
    img_dir = path.join(path.dirname(__file__), 'assets')
    sound_folder = path.join(path.dirname(__file__), 'sounds')

    ###############################

    # 初始化各项参数
    # to be placed in "constant.py" later
    WIDTH = 480
    HEIGHT = 600
    FPS = 60
    POWERUP_TIME = 5000
    BAR_LENGTH = 100
    BAR_HEIGHT = 10

    # Define Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)

    ###############################

    update_count = 0

    ###############################
    # to placed in "__init__.py" later
    # initialize pygame and create window
    pygame.init()
    pygame.mixer.init()  # For sound
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Space Shooter")
    clock = pygame.time.Clock()  # For syncing the FPS
    ###############################

    font_name = pygame.font.match_font('arial')

    def main_menu():
        # global screen

        menu_song = pygame.mixer.music.load(path.join(sound_folder, "menu.ogg"))
        pygame.mixer.music.play(-1)

        title = pygame.image.load(path.join(img_dir, "main.png")).convert()
        title = pygame.transform.scale(title, (WIDTH, HEIGHT), screen)

        screen.blit(title, (0, 0))
        pygame.display.update()

        while True:
            ev = pygame.event.poll()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    break
                elif ev.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
            elif ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            else:
                draw_text(screen, "Press [ENTER] To Begin", 30, WIDTH / 2, HEIGHT / 2)
                draw_text(screen, "or [Q] To Quit", 30, WIDTH / 2, (HEIGHT / 2) + 40)
                pygame.display.update()

        # pygame.mixer.music.stop()
        ready = pygame.mixer.Sound(path.join(sound_folder, 'getready.ogg'))
        ready.play()
        screen.fill(BLACK)
        draw_text(screen, "GET READY!", 40, WIDTH / 2, HEIGHT / 2)
        pygame.display.update()

    def draw_text(surf, text, size, x, y):
        # selecting a cross platform font to display the score
        font = pygame.font.Font(font_name, size)
        text_surface = font.render(text, True, WHITE)  ## True denotes the font to be anti-aliased
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        surf.blit(text_surface, text_rect)

    def draw_shield_bar(surf, x, y, pct):
        # if pct < 0:
        #     pct = 0
        pct = max(pct, 0)
        ## moving them to top
        # BAR_LENGTH = 100
        # BAR_HEIGHT = 10
        fill = (pct / 100) * BAR_LENGTH
        outline_rect = pygame.Rect(x, y, BAR_LENGTH, BAR_HEIGHT)
        fill_rect = pygame.Rect(x, y, fill, BAR_HEIGHT)
        pygame.draw.rect(surf, GREEN, fill_rect)
        pygame.draw.rect(surf, WHITE, outline_rect, 2)

    def draw_lives(surf, x, y, lives, img):
        for i in range(lives):
            img_rect = img.get_rect()
            img_rect.x = x + 30 * i
            img_rect.y = y
            surf.blit(img, img_rect)

    def newmob():
        mob_element = Mob()
        all_sprites.add(mob_element)
        mobs.add(mob_element)

    class Explosion(pygame.sprite.Sprite):
        # 爆炸
        def __init__(self, center, size):
            pygame.sprite.Sprite.__init__(self)
            self.size = size
            self.image = explosion_anim[self.size][0]
            self.rect = self.image.get_rect()
            self.rect.center = center
            self.frame = 0
            self.last_update = pygame.time.get_ticks()
            self.frame_rate = 75

        def update(self):
            now = pygame.time.get_ticks()
            if now - self.last_update > self.frame_rate:
                self.last_update = now
                self.frame += 1
                if self.frame == len(explosion_anim[self.size]):
                    self.kill()
                else:
                    center = self.rect.center
                    self.image = explosion_anim[self.size][self.frame]
                    self.rect = self.image.get_rect()
                    self.rect.center = center

    class Player(pygame.sprite.Sprite):
        # 飞机
        def __init__(self):
            pygame.sprite.Sprite.__init__(self)
            # scale the player img down
            self.image = pygame.transform.scale(player_img, (50, 38))
            self.image.set_colorkey(BLACK)
            self.rect = self.image.get_rect()
            self.radius = 20
            self.rect.centerx = WIDTH / 2
            self.rect.bottom = HEIGHT - 10
            self.speedx = 0
            self.speedy = 0
            self.shield = 100 # 生命值
            self.shoot_delay = 250
            self.last_shot = pygame.time.get_ticks()
            self.lives = 3
            self.hidden = False
            self.hide_timer = pygame.time.get_ticks()
            self.power = 1
            self.power_timer = pygame.time.get_ticks()

        def update(self):
            # time out for powerups
            if self.power >= 2 and pygame.time.get_ticks() - self.power_time > POWERUP_TIME:
                self.power -= 1
                self.power_time = pygame.time.get_ticks()

            # unhide
            if self.hidden and pygame.time.get_ticks() - self.hide_timer > 1000:
                self.hidden = False
                self.rect.centerx = WIDTH / 2
                self.rect.bottom = HEIGHT - 30

            self.speedx = 0.75*self.speedx  # makes the player static in the screen by default.
            self.speedy = 0.75*self.speedy
            # then we have to check whether there is an event hanlding being done for the arrow keys being
            # pressed

            # will give back a list of the keys which happen to be pressed down at that moment
            keystate = pygame.key.get_pressed()
            if keystate[pygame.K_LEFT]:
                self.speedx = -5
            elif keystate[pygame.K_RIGHT]:
                self.speedx = 5
            elif keystate[pygame.K_UP]:
                self.speedy = -5
            elif keystate[pygame.K_DOWN]:
                self.speedy = 5

            # 用对象检测和对象追踪操作player

            object_center = {}

            if not q.empty():
                object_center = q.get(True)

            if object_center:
                self.follow_object(object_center)

            '''
            # Fire weapons by holding spacebar
            if keystate[pygame.K_SPACE]:
                self.shoot()
            '''
            # fire weapons automatically
            if update_count % 2 == 1:
                self.shoot()

            # check for the borders at the left and right
            if self.rect.right > WIDTH:
                self.rect.right = WIDTH
            if self.rect.left < 0:
                self.rect.left = 0

            # check for the borders at the top and bottom
            if self.rect.top < 0:
                self.rect.top = 0
            if self.rect.bottom > HEIGHT:
                self.rect.bottom = HEIGHT

            self.rect.x += self.speedx
            self.rect.y += self.speedy

        def follow_object(self, object_center):
            distance_x = (1 - object_center['x']) * WIDTH - self.rect.x
            if abs(distance_x) != 0:
                self.speedx = 10 * distance_x / abs(distance_x)
            distance_y = object_center['y'] * HEIGHT - self.rect.y
            if abs(distance_y) != 0:
                self.speedy = 10 * distance_y / abs(distance_y)

        def shoot(self):
            # to tell the bullet where to spawn
            now = pygame.time.get_ticks()
            if now - self.last_shot > self.shoot_delay:
                self.last_shot = now
                if self.power == 1:
                    bullet = Bullet(self.rect.centerx, self.rect.top)
                    all_sprites.add(bullet)
                    bullets.add(bullet)
                    shooting_sound.play()
                if self.power == 2:
                    bullet1 = Bullet(self.rect.left, self.rect.centery)
                    bullet2 = Bullet(self.rect.right, self.rect.centery)
                    all_sprites.add(bullet1)
                    all_sprites.add(bullet2)
                    bullets.add(bullet1)
                    bullets.add(bullet2)
                    shooting_sound.play()

                """ MOAR POWAH """
                if self.power >= 3:
                    bullet1 = Bullet(self.rect.left, self.rect.centery)
                    bullet2 = Bullet(self.rect.right, self.rect.centery)
                    missile1 = Missile(self.rect.centerx, self.rect.top)  # Missile shoots from center of ship
                    all_sprites.add(bullet1)
                    all_sprites.add(bullet2)
                    all_sprites.add(missile1)
                    bullets.add(bullet1)
                    bullets.add(bullet2)
                    bullets.add(missile1)
                    shooting_sound.play()
                    missile_sound.play()

        def powerup(self):
            self.power += 1
            self.power_time = pygame.time.get_ticks()

        def hide(self):
            self.hidden = True
            self.hide_timer = pygame.time.get_ticks()
            self.rect.center = (WIDTH / 2, HEIGHT + 200)

    class Mob(pygame.sprite.Sprite):
        # 陨石
        def __init__(self):
            pygame.sprite.Sprite.__init__(self)
            self.image_orig = random.choice(meteor_images)
            self.image_orig.set_colorkey(BLACK)
            self.image = self.image_orig.copy()
            self.rect = self.image.get_rect()
            self.radius = int(self.rect.width * .90 / 2)
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(-150, -100)
            self.speedy = random.randrange(5, 20)  # for randomizing the speed of the Mob

            # randomize the movements a little more
            self.speedx = random.randrange(-3, 3)

            # adding rotation to the mob element
            self.rotation = 0
            self.rotation_speed = random.randrange(-8, 8)
            self.last_update = pygame.time.get_ticks()  # time when the rotation has to happen

        def rotate(self):
            time_now = pygame.time.get_ticks()
            if time_now - self.last_update > 50:  # in milliseconds
                self.last_update = time_now
                self.rotation = (self.rotation + self.rotation_speed) % 360
                new_image = pygame.transform.rotate(self.image_orig, self.rotation)
                old_center = self.rect.center
                self.image = new_image
                self.rect = self.image.get_rect()
                self.rect.center = old_center

        def update(self):
            self.rotate()
            self.rect.x += self.speedx
            self.rect.y += self.speedy
            # now what if the mob element goes out of the screen

            if (self.rect.top > HEIGHT + 10) or (self.rect.left < -25) or (self.rect.right > WIDTH + 20):
                self.rect.x = random.randrange(0, WIDTH - self.rect.width)
                self.rect.y = random.randrange(-100, -40)
                self.speedy = random.randrange(1, 8)  ## for randomizing the speed of the Mob

    # defines the sprite for Powerups
    class Pow(pygame.sprite.Sprite):
        # 奖励物
        def __init__(self, center):
            pygame.sprite.Sprite.__init__(self)
            self.type = random.choice(['shield', 'gun'])
            self.image = powerup_images[self.type]
            self.image.set_colorkey(BLACK)
            self.rect = self.image.get_rect()
            # place the bullet according to the current position of the player
            self.rect.center = center
            self.speedy = 2

        def update(self):
            """should spawn right in front of the player"""
            self.rect.y += self.speedy
            # kill the sprite after it moves over the top border
            if self.rect.top > HEIGHT:
                self.kill()

    # defines the sprite for bullets
    class Bullet(pygame.sprite.Sprite):
        # 子弹
        def __init__(self, x, y):
            pygame.sprite.Sprite.__init__(self)
            self.image = bullet_img
            self.image.set_colorkey(BLACK)
            self.rect = self.image.get_rect()
            # place the bullet according to the current position of the player
            self.rect.bottom = y
            self.rect.centerx = x
            self.speedy = -10

        def update(self):
            """should spawn right in front of the player"""
            self.rect.y += self.speedy
            # kill the sprite after it moves over the top border
            if self.rect.bottom < 0:
                self.kill()

            ## now we need a way to shoot
            ## lets bind it to "spacebar".
            ## adding an event for it in Game loop

    # FIRE ZE MISSILES
    class Missile(pygame.sprite.Sprite):
        # 导弹
        def __init__(self, x, y):
            pygame.sprite.Sprite.__init__(self)
            self.image = missile_img
            self.image.set_colorkey(BLACK)
            self.rect = self.image.get_rect()
            self.rect.bottom = y
            self.rect.centerx = x
            self.speedy = -10

        def update(self):
            """should spawn right in front of the player"""
            self.rect.y += self.speedy
            if self.rect.bottom < 0:
                self.kill()

    ###################################################

    # Load all game images

    background = pygame.image.load(path.join(img_dir, 'starfield.png')).convert()
    background_rect = background.get_rect()
    # ^^ draw this rect first

    player_img = pygame.image.load(path.join(img_dir, 'playerShip1_orange.png')).convert()
    player_mini_img = pygame.transform.scale(player_img, (25, 19))
    player_mini_img.set_colorkey(BLACK)
    bullet_img = pygame.image.load(path.join(img_dir, 'laserRed16.png')).convert()
    missile_img = pygame.image.load(path.join(img_dir, 'missile.png')).convert_alpha()
    # meteor_img = pygame.image.load(path.join(img_dir, 'meteorBrown_med1.png')).convert()
    meteor_images = []
    meteor_list = [
        'meteorBrown_big1.png',
        'meteorBrown_big2.png',
        'meteorBrown_med1.png',
        'meteorBrown_med3.png',
        'meteorBrown_small1.png',
        'meteorBrown_small2.png',
        'meteorBrown_tiny1.png'
    ]

    for image in meteor_list:
        meteor_images.append(pygame.image.load(path.join(img_dir, image)).convert())

    # meteor explosion
    explosion_anim = dict()
    explosion_anim['lg'] = []
    explosion_anim['sm'] = []
    explosion_anim['player'] = []
    for i in range(9):
        filename = 'regularExplosion0{}.png'.format(i)
        img = pygame.image.load(path.join(img_dir, filename)).convert()
        img.set_colorkey(BLACK)
        # resize the explosion
        img_lg = pygame.transform.scale(img, (75, 75))
        explosion_anim['lg'].append(img_lg)
        img_sm = pygame.transform.scale(img, (32, 32))
        explosion_anim['sm'].append(img_sm)

        # player explosion
        filename = 'sonicExplosion0{}.png'.format(i)
        img = pygame.image.load(path.join(img_dir, filename)).convert()
        img.set_colorkey(BLACK)
        explosion_anim['player'].append(img)

    # load power ups
    powerup_images = dict()
    powerup_images['shield'] = pygame.image.load(path.join(img_dir, 'shield_gold.png')).convert()
    powerup_images['gun'] = pygame.image.load(path.join(img_dir, 'bolt_gold.png')).convert()

    ###################################################

    ###################################################

    # Load all game sounds
    shooting_sound = pygame.mixer.Sound(path.join(sound_folder, 'pew.wav'))
    missile_sound = pygame.mixer.Sound(path.join(sound_folder, 'rocket.ogg'))
    expl_sounds = []
    for sound in ['expl3.wav', 'expl6.wav']:
        expl_sounds.append(pygame.mixer.Sound(path.join(sound_folder, sound)))
    # main background music
    # pygame.mixer.music.load(path.join(sound_folder, 'tgfcoder-FrozenJam-SeamlessLoop.ogg'))
    pygame.mixer.music.set_volume(0.2)  # simmered the sound down a little

    player_die_sound = pygame.mixer.Sound(path.join(sound_folder, 'rumble1.ogg'))
    ###################################################

    # TODO: make the game music loop over again and again. play(loops=-1) is not working
    # Error :
    # TypeError: play() takes no keyword arguments
    # pygame.mixer.music.play()

    #############################
    # Game loop
    running = True
    menu_display = True
    while running:
        if menu_display:
            main_menu()
            pygame.time.wait(3000)

            # Stop menu music
            pygame.mixer.music.stop()
            # Play the gameplay music
            pygame.mixer.music.load(path.join(sound_folder, 'tgfcoder-FrozenJam-SeamlessLoop.ogg'))
            pygame.mixer.music.play(-1)  # makes the gameplay sound in an endless loop

            menu_display = False

            # group all the sprites together for ease of update
            all_sprites = pygame.sprite.Group()
            player = Player()
            all_sprites.add(player)

            # spawn a group of mob
            mobs = pygame.sprite.Group()
            for i in range(8):  # 8 mobs
                newmob()

            # group for bullets
            bullets = pygame.sprite.Group()
            powerups = pygame.sprite.Group()

            # Score board variable
            score = 0

        # 1 Process input/events
        clock.tick(FPS)  # will make the loop run at the same speed all the time
        for event in pygame.event.get():  # gets all the events which have occured till now and keeps tab of them.
            # listening for the the X button at the top
            if event.type == pygame.QUIT:
                running = False

            # Press ESC to exit game
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            # ## event for shooting the bullets
            # elif event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         player.shoot()      ## we have to define the shoot()  function

        # 2 Update
        all_sprites.update()
        update_count += 1

        # check if a bullet hit a mob
        # now we have a group of bullets and a group of mob
        hits = pygame.sprite.groupcollide(mobs, bullets, True, True)
        # now as we delete the mob element when we hit one with a bullet, we need to respawn them again
        # as there will be no mob_elements left out
        for hit in hits:
            score += 50 - hit.radius  # give different scores for hitting big and small metoers
            random.choice(expl_sounds).play()
            expl = Explosion(hit.rect.center, 'lg')
            all_sprites.add(expl)
            if random.random() > 0.9:
                pow = Pow(hit.rect.center)
                all_sprites.add(pow)
                powerups.add(pow)
            newmob()  # spawn a new mob

        # ^^ the above loop will create the amount of mob objects which were killed spawn again
        #########################

        # check if the player collides with the mob
        hits = pygame.sprite.spritecollide(player, mobs, True,
                                           pygame.sprite.collide_circle)
        # gives back a list, True makes the mob element disappear

        for hit in hits:
            player.shield -= hit.radius * 2
            expl = Explosion(hit.rect.center, 'sm')
            all_sprites.add(expl)
            newmob()
            if player.shield <= 0:
                player_die_sound.play()
                death_explosion = Explosion(player.rect.center, 'player')
                all_sprites.add(death_explosion)
                # running = False     ## GAME OVER 3:D
                player.hide()
                player.lives -= 1
                player.shield = 100

        # if the player hit a power up
        hits = pygame.sprite.spritecollide(player, powerups, True)
        for hit in hits:
            if hit.type == 'shield':
                player.shield += random.randrange(10, 30)
                if player.shield >= 100:
                    player.shield = 100
            if hit.type == 'gun':
                player.powerup()

        # if player died and the explosion has finished, end game
        if player.lives == 0 and not death_explosion.alive():
            running = False
            # menu_display = True
            # pygame.display.update()

        # 3 Draw/render
        screen.fill(BLACK)
        # draw the stargaze.png image
        screen.blit(background, background_rect)

        all_sprites.draw(screen)
        draw_text(screen, str(score), 18, WIDTH / 2, 10)  # 10px down from the screen
        draw_shield_bar(screen, 5, 5, player.shield)

        # Draw lives
        draw_lives(screen, WIDTH - 100, 5, player.lives, player_mini_img)

        # Done after drawing everything to the screen
        pygame.display.flip()

    sys.exit()


set_process = multiprocessing.Process(target=set_object_center)
game_process = multiprocessing.Process(target=game)

set_process.start()
game_process.start()

game_process.join()
set_process.terminate()

print("退出主线程")