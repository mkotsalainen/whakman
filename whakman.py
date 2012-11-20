#!/usr/bin/env python

import random, os.path
import pygame
import math
from pygame.locals import *
from heapq import heappush, heappop

# Global variables
SCREENRECT= Rect(0, 0, 640, 640)
IMAGECACHE = {}
WHITE = (255,255,255)
BLACK = (0, 0, 0)
SCREEN = None
KEYSTATE = None
CLOCK = None
TILE_SIZE = 64
TILE_ROWS = 10
TILE_COLS = 10
MAP_OFFSET_X = TILE_SIZE/2    #at what pos the map is placed
MAP_OFFSET_Y = TILE_SIZE/2

N = 0
E = 1
S = 2
W = 3

# Toggle this to see A* algorithm in action
VISUALIZE_PATHFINDING = False

# Load an image from a file, with possible rotation and/or flip
def image(file, rotation = 0, flipx = False, flipy = False):
  key = (file, rotation, flipx, flipy)
  if not IMAGECACHE.has_key(key):
    path = os.path.join('images', file)
    ext = ["", ".bmp", ".gif", ".png"]
    for e in ext:
      if os.path.exists(path + e):
        path = path + e
        break
    if rotation or flipx or flipy:
      img = image(file)
    else:
      img = pygame.image.load(path).convert_alpha()
    if rotation:
      img = pygame.transform.rotate(img, rotation)
    if flipx or flipy:
      img = pygame.transform.flip(img, flipx, flipy)
    IMAGECACHE[key] = img
  return IMAGECACHE[key]



############################################
# Vec
############################################
class Vec:

  def __init__(self, x, y):
    self.x, self.y = x, y

  def mult(self, val):
    return Vec(self.x*val, self.y*val)

  def scale(self, val):
    l = self.length()
    if l > 0: return Vec(self.x*val/l, self.y*val/l)
    else: return Vec(0,0)

  def __add__(self, other):
    return Vec(self.x+other.x, self.y+other.y)

  def __sub__(self, other):
    return Vec(self.x-other.x, self.y-other.y)

  def __repr__(self):
    return '(%f,%f)' % (self.x, self.y)

  def length(self):
    return math.sqrt(self.x*self.x + self.y*self.y)

  def length_sq(self):
    return self.x*self.x + self.y*self.y

  def neg(self):
    return Vec(self.x*-1, self.y*-1)

  def n(self):
    l = self.length()
    if l > 0: return Vec(self.x/l, self.y/l)
    else: return Vec(0,0)

  def dot(self, other):
    return self.x*other.x + self.y*other.y

  def clear(self):
    self.x = self.y = 0

  def is_zero(self):
    return abs(self.x) < 0.0000001 and abs(self.y) < 0.0000001

  def copy(self):
    return Vec(self.x, self.y)

  def to_dir(self):
    #translates a (-1,0) etc vector to a dir
    #game entities move horiz or vertically

    if abs(self.x) + abs(self.y) != 1:
     raise "can't turn vec into dir: %s" % self

    if abs(self.y):
      if self.y < 0: return N
      else: return S
    else:
      if self.x > 0: return E
      else: return W



############################################
# PathNode - used for ghost pathfinding
############################################
class PathNode:

  def __init__(self, tile):
    self.tile = tile
    self.cost_so_far = 0
    self.estimated_total_cost = 0     #cost to get to this node from start + estimate of cost to get to goal from here
    self.in_path = 0  #for debugging

  def neighbors(self):
    return [tile.node for tile in self.tile.neighbors.values()]

  def cost_heuristic(self, goal):
    d = goal.tile.pos - self.tile.pos
    return (abs(d.x) + abs(d.y))/TILE_SIZE

  def __eq__(self, other):
    return self.tile == other.tile

  def __cmp__(self, other):
    return self.estimated_total_cost - other.estimated_total_cost

  def __repr__(self):
    return '%f %s' % (self.cost_so_far, self.tile)



############################################
# Tile - the map is made up of square tiles connected with each other
############################################
class Tile:

  def __init__(self, type, row, col):
    self.type = type
    self.row = row
    self.col = col
    self.node = PathNode(self)
    self.contains_glove = (self.row == 9 or self.row == 0) and (self.col == 0 or self.col == 9)
    self.contains_coin = ((self.col + self.row) % 3 == 0)
    if not self.type.image: return
    self.rect = self.type.image.get_rect()
    self.rect.centerx = MAP_OFFSET_X + col*TILE_SIZE
    self.rect.centery = MAP_OFFSET_Y + row*TILE_SIZE
    self.pos = Vec(self.rect.centerx, self.rect.centery)

  def cache_neighbors(self):
    self.neighbors = {}
    if self.type.has_edge_towards(N): self.neighbors[N] = MAP.tile(self.row - 1, self.col)
    if self.type.has_edge_towards(E) and self.col < TILE_COLS-1: self.neighbors[E] = MAP.tile(self.row, self.col + 1)
    if self.type.has_edge_towards(S): self.neighbors[S] = MAP.tile(self.row + 1, self.col)
    if self.type.has_edge_towards(W) and self.col > 0: self.neighbors[W] = MAP.tile(self.row, self.col - 1)

  #returns the tiles' neighbor in a givien direction, or None
  def neighbor(self, dir):
    if self.neighbors.has_key(dir):
      return self.neighbors[dir]

  #triggered by whakman by walking on tile
  def hit(self):
    self.contains_coin = False
    if self.contains_glove:
      self.contains_glove = False
      for ghost in GHOSTS:
        ghost.scare()

  def draw(self):
    if not self.type.image: return
    SCREEN.blit(self.type.image, self.rect)
    if not VISUALIZE_PATHFINDING:
      if self.contains_coin:
        SCREEN.blit(image("rock"), self.rect)
      if self.contains_glove:
          SCREEN.blit(image("glove"), self.rect)
    if VISUALIZE_PATHFINDING and self.node.in_path:
      SCREEN.blit(image("bomb"), self.rect)

  def __repr__(self):
    return 'tile (row, col): (%d,%d)' % (self.row, self.col)



############################################
# TileType - there are a few different types, t junctions, crosses etc
############################################

class TileType:

  def __init__(self, image, edges):
    self.image = image
    self.edges = edges

  def rotate(self, n):
    new_image = pygame.transform.rotate(self.image, 90*n)
    new_edges = [self.edges[(0+n)%4], self.edges[(1+n)%4], self.edges[(2+n)%4], self.edges[(3+n)%4]]
    return TileType(new_image, new_edges)

  def has_edge_towards(self, dir):

    return self.edges[dir] == 1



############################################
# Map - a wrapper for a 2d array of tiles
############################################

class Map:

  def __init__(self):
    self.create_tile_types()

  def create_tile_types(self):
    empty = TileType(None, [0,0,0,0])
    cross = TileType(image('wall_cross'), [1, 1, 1, 1])
    turn = TileType(image('wall_turn'), [0, 1, 1, 0])
    t = TileType(image('wall_t'), [1, 0, 1, 1])
    straight = TileType(image('wall_straight'), [0, 1, 0, 1])
    end = TileType(image('wall_end'), [1, 0, 0, 0])
    self.tile_types = [empty, cross, turn, turn.rotate(1), turn.rotate(2), turn.rotate(3), t, t.rotate(1), t.rotate(2), t.rotate(3), straight, straight.rotate(1), end, end.rotate(1), end.rotate(2), end.rotate(3)]

  def create_level(self, level_file):
    data = [line.strip() for line in open(level_file).readlines()]  #load data from level file and put it into an array of strings
    self.level = [[None]*TILE_COLS for j in range(TILE_ROWS)] #create empty multidim array
    self.tiles = [] #for easy sequential access
    for row in range(TILE_ROWS):
      for col in range(TILE_COLS):
        tile_index = int(data[row][col], 16)
        tile = Tile(self.tile_types[tile_index], row, col)
        self.level[row][col] = tile
        self.tiles.append(tile)

    for tile in self.tiles:
      tile.cache_neighbors()

  def tile(self, row, col):
    return self.level[row][col]

  def draw(self):
    for row in range(TILE_ROWS):
      for col in range(TILE_COLS):
        self.level[row][col].draw()



############################################
# Whakman
############################################

class Whakman:

  def __init__(self, images, target):
    self.images = images
    self.rect = images[0].get_rect()
    self.mouthtimer = 0
    self.image = images[0]
    self.last_blit_image = images[0]
    self.pos = target.pos
    self.vel = Vec(0,0)
    self.last_pressed_keys = Vec(0,0)
    self.speed = 8.0  #less is faster
    self.target = target
    mix_in(Whakman, Floater)
    self.init_float(30.0, 1)

  def update(self):
    self.update_float()
    self.update_keys()
    if self.last_pressed_keys.is_zero():  #this will be the case before we make our first keypress
      return
    self.update_pos()
    self.update_mouth()

  def update_keys(self):
    keys = Vec(0,0)
    if KEYSTATE[K_UP]: keys.y -= 1
    if KEYSTATE[K_DOWN]: keys.y += 1
    if KEYSTATE[K_LEFT]: keys.x -= 1
    if KEYSTATE[K_RIGHT]: keys.x += 1

    #give precedence to vertical if we're travelling horizontally & vice versa
    if abs(keys.x) and abs(keys.y):
      if abs(self.vel.x): keys.x = 0
      else: keys.y = 0

    #turn on a dime
    if self.vel.x and keys.x and self.vel.x != keys.x:
      self.turn_back()
    if self.vel.y and keys.y and self.vel.y != keys.y:
        self.turn_back()

    if not keys.is_zero():
      self.last_pressed_keys = keys

  def turn_back(self):
    self.target, self.old_target = self.old_target, self.target
    self.vel = self.vel.neg()

  def update_pos(self):
    d = self.target.pos - self.pos
    at_target = (d.length_sq() < 5.0)     #TODO: we could overshoot if update is delayed, maybe clamp CLOCK
    if at_target:
      self.target.hit()
      self.pos = self.target.pos
      new_target = self.target.neighbor(self.last_pressed_keys.to_dir())
      if new_target:
        self.vel = self.last_pressed_keys.copy()
        self.old_target = self.target
        self.target = new_target
      else:
        self.vel.clear() #we're stuck
    else:
      ts = CLOCK.get_time()/self.speed
      self.pos += self.vel.mult(ts)

  def update_mouth(self):
    if self.vel.is_zero(): return
    self.mouthtimer += CLOCK.get_time()
    if self.mouthtimer > 300:
      self.flip_mouth()
      self.mouthtimer = 0

  def flip_mouth(self):
    if self.image == self.images[0]: self.image = self.images[1]
    else: self.image = self.images[0]

  def draw(self):
    if self.vel.y < 0: blit_image = pygame.transform.rotate(self.image, 90)
    elif self.vel.y > 0: blit_image = pygame.transform.rotate(self.image, -90)
    elif self.vel.x < 0: blit_image = pygame.transform.flip(self.image, 1, 0)
    elif self.vel.x > 0: blit_image = self.image
    else: blit_image = self.last_blit_image

    self.rect.centerx = self.pos.x + self.float_pos.x
    self.rect.centery = self.pos.y + self.float_pos.y
    SCREEN.blit(blit_image, self.rect)
    self.last_blit_image = blit_image


############################################
# Floater - a mixin that adds some "noise" movement to game entities
############################################

class Floater:

  #speed -> faster is slower
  #float_r - the offset will be chosen randomly on a circle with this radius
  def init_float(self, float_speed, float_r):
    self.float_r = float_r
    self.float_speed = float_speed
    self.float_pos = Vec(0,0)
    self.find_new_float_target()

  def find_new_float_target(self):
    a = random.uniform(0, 2*math.pi)
    self.float_target_pos = Vec(math.cos(a), math.sin(a)).mult(self.float_r)

  def update_float(self):
    d = self.float_target_pos - self.float_pos
    if (d.length_sq() < 2):
      self.find_new_float_target()
    self.float_pos += d.scale(CLOCK.get_time()/self.float_speed)

############################################
# Ghost
############################################

class Ghost(Floater):

  def __init__(self, normal_image, scared_image, target):
    self.rect = normal_image.get_rect()
    self.normal_image = normal_image
    self.scared_image = scared_image
    self.pos = target.pos
    self.vel = Vec(0,0)
    self.speed = 8.0
    self.target = target
    self.scared_timer = 0
    self.find_new_target()
    mix_in(Ghost, Floater)
    self.init_float(120.0, 10)

  def update(self):
    self.update_float()
    if self.scared_timer > 0:   #we only want to stay scared for a while
      self.scared_timer -= CLOCK.get_time()
    d = self.target.pos - self.pos
    if d.length_sq() < 2.0:
      self.pos = self.target.pos
      self.find_new_target()
    else:
      self.vel = d.scale(CLOCK.get_time() / self.speed)
      self.pos += self.vel

  def scare(self):
    self.scared_timer = 4000;

  def find_new_target(self):
    if self.target == WHAKMAN.target:
      return
    path = self.find_path(self.target.node, WHAKMAN.target.node)
    if self.scared_timer > 0:
      #choose any valid path except for the one that leades to whakman
      self.find_new_random_target(path[0].tile)
    else:
      self.target = path[0].tile

  def find_new_random_target(self, not_allowed):
    new_target = None
    while not new_target:
        new_target = self.target.neighbor(random.randrange(4))
        if new_target and new_target != not_allowed:
          self.target = new_target

  def draw(self):
    self.rect.centerx = self.pos.x + self.float_pos.x
    self.rect.centery = self.pos.y + self.float_pos.y
    if self.scared_timer > 0: blit_image = self.scared_image
    else: blit_image = self.normal_image
    if self.vel.x >= 0: blit_image = pygame.transform.flip(blit_image, 1, 0)
    SCREEN.blit(blit_image, self.rect)

  #returns an array containing all nodes between start and goal
  #for this level, we will always be able to find the goal
  def find_path(self, start, goal):

    for tile in MAP.tiles:
      tile.node.cost_so_far = 0
      tile.node.in_path = 0

    start.estimated_total_cost = start.cost_heuristic(goal)
    open = [start]
    closed = []

    count = 0
    while len(open) > 0:
      count += 1
      current = open[0]
      if current == goal: break

      for neighbor in current.neighbors():
        if neighbor in closed:
          if neighbor.cost_so_far <= current.cost_so_far + 1:
            continue  #skip, we've already got a better path
          else:
            closed.remove(neighbor) #reopen the node, we can do better

        elif neighbor in open:
          if neighbor.cost_so_far <= current.cost_so_far + 1:
            continue  #skip, we've already got a better path

        neighbor.cost_so_far = current.cost_so_far + 1
        neighbor.estimated_total_cost = neighbor.cost_so_far + neighbor.cost_heuristic(goal)
        neighbor.parent = current
        if not neighbor in open:
          heappush(open, neighbor)
      heappop(open) #we're done with current, remove it from top of heap
      closed.append(current)

    #when we have exausted the open list, we've got the goal
    path = []
    while not current == start:   #This breaks if I do while current != start. Weird
      path.append(current)
      current.in_path = True
      current = current.parent
    path.reverse()
    return path


def mix_in(pyClass, mixInClass, makeLast=0):
  if mixInClass not in pyClass.__bases__:
    pyClass.__bases__ += (mixInClass,)




def main():
  global SCREEN, CLOCK, KEYSTATE, MAP, WHAKMAN, GHOSTS

  # Initialize pygame
  pygame.init()
  pygame.mixer.get_init()
  bestdepth = pygame.display.mode_ok(SCREENRECT.size, pygame.DOUBLEBUF, 32)
  SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, bestdepth)
  pygame.display.set_caption('Whakman')
  CLOCK = pygame.time.Clock()

  MAP = Map()
  MAP.create_level("level0.txt")

  # Create some actors
  WHAKMAN = Whakman([image('whakman_01'), image('whakman_02')], MAP.tile(7,5))
  ghost1 = Ghost(image('ghost_01'), image('ghost_02'), MAP.tile(4,4))
  GHOSTS = [ghost1]

  drawlist = [MAP, WHAKMAN] + GHOSTS
  updatelist = [WHAKMAN] + GHOSTS

  # Main loop
  while True:
    for event in pygame.event.get():
      if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
          return
    KEYSTATE = pygame.key.get_pressed()

    if CLOCK.get_time() > 0:
      for a in updatelist: a.update()

    SCREEN.fill(BLACK)
    for a in drawlist: a.draw()
    pygame.display.flip()

    CLOCK.tick(60)

main()

