# True  = train the AI
# False = test the trained AI without exploration

TRAINING_MODE = True # For training the AI

#TRAINING_MODE = False # For testing the AI

# ============================================================
# PYGAME INITIALIZATION
# ============================================================

pygame.init()

WIDTH = 600
HEIGHT = 600

GRID_SIZE = 10
CELL_SIZE = WIDTH // GRID_SIZE

FPS = 60

win = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Snake Q-Learning AI V2"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont(
    "Arial",
    18,
    bold=True
)

small_font = pygame.font.SysFont(
    "Arial",
    14
)


# ============================================================
# COLORS
# ============================================================

BACKGROUND = (235, 242, 235)
GRID_COLOR = (190, 205, 190)

SNAKE_HEAD = (35, 180, 70)
SNAKE_BODY = (50, 155, 65)
SNAKE_DARK = (25, 110, 45)

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)

FOOD_RED = (220, 45, 45)
FOOD_HIGHLIGHT = (255, 150, 150)

EYE_COLOR = (10, 10, 10)

LEAF_GREEN = (45, 130, 55)
STEM_BROWN = (100, 60, 25)


# ============================================================
# ACTIONS
# ============================================================

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

ACTIONS = [
    UP,
    DOWN,
    LEFT,
    RIGHT
]


# ============================================================
# FILES
# ============================================================

EXCEL_FILE = (
    "snake_q_learning_training.xlsx"
)

Q_TABLE_FILE = (
    "snake_q_table.npy"
)


# ============================================================
# Q-LEARNING PARAMETERS
# ============================================================

alpha = 0.10

gamma = 0.90

# Initial exploration
epsilon = 1.0

# Minimum exploration
epsilon_min = 0.05

# Exploration decay
epsilon_decay = 0.995

# Episodes per training run
num_episodes = 5000

# Maximum steps per episode
# MAX_STEPS = 500
MAX_STEPS = 2000

# ============================================================
# FOOD
# ============================================================

class Food:

    def __init__(self):
        self.position = (
            0,
            0
        )

    def get_random_position(self):

        return (
            random.randint(
                0,
                GRID_SIZE - 1
            ),
            random.randint(
                0,
                GRID_SIZE - 1
            )
        )

    def randomize(self, snake=None):

        # Try to find a free position

        if snake is None:

            self.position = (
                self.get_random_position()
            )

            return


        free_positions = []

        for x in range(GRID_SIZE):

            for y in range(GRID_SIZE):

                position = (
                    x,
                    y
                )

                if position not in snake.positions:

                    free_positions.append(
                        position
                    )


        if free_positions:

            self.position = random.choice(
                free_positions
            )


# ============================================================
# SNAKE
# ============================================================

class Snake:

    def __init__(self):

        center = GRID_SIZE // 2

        self.positions = [

            (
                center,
                center
            )
        ]

        self.direction = random.choice(
            ACTIONS
        )


    def move(self, action):

        self.direction = action

        head_x, head_y = (
            self.positions[0]
        )

        dir_x, dir_y = action

        new_head = (

            head_x + dir_x,

            head_y + dir_y
        )

        self.positions = [

            new_head

        ] + self.positions[:-1]


    def grow(self):

        tail = self.positions[-1]

        self.positions.append(
            tail
        )


    def collision(self):

        head = self.positions[0]

        # Wall

        if (

            head[0] < 0

            or head[0] >= GRID_SIZE

            or head[1] < 0

            or head[1] >= GRID_SIZE
        ):

            return True


        # Body

        if head in self.positions[1:]:

            return True


        return False


# ============================================================
# OPPOSITE ACTION
# ============================================================

def is_opposite(
    action,
    direction
):

    return (

        action[0] == -direction[0]

        and

        action[1] == -direction[1]
    )


# ============================================================
# VALID ACTIONS
# ============================================================

def get_valid_actions(snake):

    valid_actions = []

    head_x, head_y = (
        snake.positions[0]
    )

    for action_index, action in enumerate(
        ACTIONS
    ):

        # Prevent immediate reverse

        if is_opposite(
            action,
            snake.direction
        ):

            continue


        dx, dy = action

        new_head = (

            head_x + dx,

            head_y + dy
        )


        # Wall

        if (

            new_head[0] < 0

            or new_head[0] >= GRID_SIZE

            or new_head[1] < 0

            or new_head[1] >= GRID_SIZE
        ):

            continue


        # Body

        if new_head in snake.positions[:-1]:

            continue


        valid_actions.append(
            action_index
        )


    return valid_actions


# ============================================================
# STATE REPRESENTATION
# ============================================================
#
# State contains:
#
# 1. Danger straight
# 2. Danger left
# 3. Danger right
#
# 4. Moving up
# 5. Moving down
# 6. Moving left
# 7. Moving right
#
# 8. Food left
# 9. Food right
# 10. Food up
# 11. Food down
#
# 12. Tail left
# 13. Tail right
# 14. Tail up
# 15. Tail down
#
# 16. Free space straight
# 17. Free space left
# 18. Free space right
#
# 18 binary features
#
# 2^18 = 262,144 states
#
# 4 actions
#
# ============================================================

STATE_BITS = 18

STATE_SIZE = 2 ** STATE_BITS

q_table_shape = (
    STATE_SIZE,
    4
)


# ============================================================
# STATE INDEX
# ============================================================

def state_to_index(state):

    index = 0

    for value in state:

        index = (
            index * 2
            + int(value)
        )

    return index


# ============================================================
# DANGER CHECK
# ============================================================

def danger_at(
    snake,
    position
):

    x, y = position

    # Wall

    if (

        x < 0

        or x >= GRID_SIZE

        or y < 0

        or y >= GRID_SIZE
    ):

        return 1


    # Body

    if position in snake.positions[1:]:

        return 1


    return 0


# ============================================================
# STATE
# ============================================================

def get_state(
    snake,
    food
):

    head_x, head_y = (
        snake.positions[0]
    )

    food_x, food_y = (
        food.position
    )

    dx, dy = snake.direction


    # --------------------------------------------------------
    # CURRENT DIRECTION
    # --------------------------------------------------------

    moving_up = int(
        dy == -1
    )

    moving_down = int(
        dy == 1
    )

    moving_left = int(
        dx == -1
    )

    moving_right = int(
        dx == 1
    )


    # --------------------------------------------------------
    # FOOD DIRECTION
    # --------------------------------------------------------

    food_left = int(
        food_x < head_x
    )

    food_right = int(
        food_x > head_x
    )

    food_up = int(
        food_y < head_y
    )

    food_down = int(
        food_y > head_y
    )


    # --------------------------------------------------------
    # STRAIGHT
    # --------------------------------------------------------

    straight_position = (

        head_x + dx,

        head_y + dy
    )


    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    left_direction = (

        dy,

        -dx
    )

    left_position = (

        head_x
        + left_direction[0],

        head_y
        + left_direction[1]
    )


    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    right_direction = (

        -dy,

        dx
    )

    right_position = (

        head_x
        + right_direction[0],

        head_y
        + right_direction[1]
    )


    danger_straight = danger_at(
        snake,
        straight_position
    )

    danger_left = danger_at(
        snake,
        left_position
    )

    danger_right = danger_at(
        snake,
        right_position
    )


    # --------------------------------------------------------
    # TAIL POSITION
    # --------------------------------------------------------

    tail_x, tail_y = (
        snake.positions[-1]
    )

    tail_left = int(
        tail_x < head_x
    )

    tail_right = int(
        tail_x > head_x
    )

    tail_up = int(
        tail_y < head_y
    )

    tail_down = int(
        tail_y > head_y
    )


    # --------------------------------------------------------
    # FREE SPACE
    # --------------------------------------------------------

    free_straight = int(
        danger_straight == 0
    )

    free_left = int(
        danger_left == 0
    )

    free_right = int(
        danger_right == 0
    )


    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    state = (

        danger_straight,
        danger_left,
        danger_right,

        moving_up,
        moving_down,
        moving_left,
        moving_right,

        food_left,
        food_right,
        food_up,
        food_down,

        tail_left,
        tail_right,
        tail_up,
        tail_down,

        free_straight,
        free_left,
        free_right
    )


    return state


# ============================================================
# LOAD Q-TABLE
# ============================================================

def load_q_table():

    if Path(
        Q_TABLE_FILE
    ).exists():

        try:

            q_table = np.load(
                Q_TABLE_FILE
            )

            if q_table.shape == q_table_shape:

                print(
                    "\nPrevious AI memory loaded!"
                )

                print(
                    f"Q-table shape: "
                    f"{q_table.shape}"
                )

                return q_table


            print(
                "Existing Q-table has "
                "incompatible shape."
            )

            print(
                "Creating new Q-table."
            )


        except Exception as e:

            print(
                f"Error loading Q-table: {e}"
            )


    print(
        "\nNo compatible previous "
        "AI memory found."
    )

    print(
        "Creating new Q-table..."
    )


    return np.zeros(
        q_table_shape,
        dtype=np.float32
    )


# ============================================================
# SAVE Q-TABLE
# ============================================================

def save_q_table(
    q_table
):

    np.save(
        Q_TABLE_FILE,
        q_table
    )


# ============================================================
# REWARD
# ============================================================

def get_reward(
    snake,
    food,
    old_distance
):

    # --------------------------------------------------------
    # DEATH
    # --------------------------------------------------------

    if snake.collision():

        return -100


    # --------------------------------------------------------
    # FOOD
    # --------------------------------------------------------

    if (

        snake.positions[0]

        ==

        food.position
    ):

        return 100


    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    head_x, head_y = (
        snake.positions[0]
    )

    food_x, food_y = (
        food.position
    )

    new_distance = (

        abs(
            head_x - food_x
        )

        +

        abs(
            head_y - food_y
        )
    )


    # Getting closer

    if new_distance < old_distance:

        return 3


    # Moving away

    if new_distance > old_distance:

        return -2


    # No progress

    return -1


# ============================================================
# EXCEL HEADERS
# ============================================================

def create_excel_headers(ws):

    ws["A1"] = (
        "Snake Q-Learning Training Log"
    )

    ws["A1"].font = Font(
        size=20,
        bold=True
    )

    ws["A1"].alignment = Alignment(
        horizontal="center"
    )

    ws.merge_cells(
        "A1:J1"
    )


    headers = [

        "Episode",
        "Total Reward",
        "Steps",
        "Food Eaten",
        "Snake Length",

        "Alpha",
        "Gamma",
        "Epsilon",

        "Average Reward",

        "Status"
    ]


    for col, header in enumerate(
        headers,
        start=1
    ):

        cell = ws.cell(
            row=3,
            column=col,
            value=header
        )

        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

        cell.alignment = Alignment(
            horizontal="center"
        )

        cell.fill = PatternFill(
            fill_type="solid",
            fgColor="1F4E78"
        )


    widths = {

        "A": 12,
        "B": 18,
        "C": 12,
        "D": 14,
        "E": 16,

        "F": 12,
        "G": 12,
        "H": 12,

        "I": 18,
        "J": 15
    }


    for column, width in widths.items():

        ws.column_dimensions[
            column
        ].width = width


    ws.freeze_panes = "A4"


# ============================================================
# EXCEL SETUP
# ============================================================

def setup_excel():

    file_path = Path(
        EXCEL_FILE
    )


    if file_path.exists():

        wb = load_workbook(
            EXCEL_FILE
        )

        if (
            "Training Log"
            in
            wb.sheetnames
        ):

            ws = wb[
                "Training Log"
            ]

        else:

            ws = wb.create_sheet(
                "Training Log"
            )

            create_excel_headers(
                ws
            )


    else:

        wb = Workbook()

        ws = wb.active

        ws.title = (
            "Training Log"
        )

        create_excel_headers(
            ws
        )


    return wb, ws


# ============================================================
# LAST EPISODE
# ============================================================

def get_last_episode(ws):

    last_episode = 0


    for row in range(
        4,
        ws.max_row + 1
    ):

        value = ws.cell(
            row=row,
            column=1
        ).value


        if isinstance(
            value,
            (int, float)
        ):

            last_episode = max(

                last_episode,

                int(value)
            )


    return last_episode


# ============================================================
# REWARD HISTORY
# ============================================================

def load_reward_history(ws):

    history = []


    for row in range(
        4,
        ws.max_row + 1
    ):

        value = ws.cell(
            row=row,
            column=2
        ).value


        if isinstance(
            value,
            (int, float)
        ):

            history.append(
                value
            )


    return history


# ============================================================
# EXCEL UPDATE
# ============================================================

def update_excel(

    wb,
    ws,

    episode,
    total_reward,
    steps,
    food_eaten,
    snake_length,

    alpha,
    gamma,
    epsilon,

    average_reward,
    status

):

    row = ws.max_row + 1


    values = [

        episode,
        total_reward,
        steps,
        food_eaten,
        snake_length,

        alpha,
        gamma,
        epsilon,

        average_reward,

        status
    ]


    border = Border(

        left=Side(
            style="thin",
            color="D9E1F2"
        ),

        right=Side(
            style="thin",
            color="D9E1F2"
        ),

        top=Side(
            style="thin",
            color="D9E1F2"
        ),

        bottom=Side(
            style="thin",
            color="D9E1F2"
        )
    )


    for col, value in enumerate(
        values,
        start=1
    ):

        cell = ws.cell(
            row=row,
            column=col,
            value=value
        )

        cell.alignment = Alignment(
            horizontal="center"
        )

        cell.border = border


    for col in [
        6,
        7,
        8,
        9
    ]:

        ws.cell(
            row=row,
            column=col
        ).number_format = (
            "0.00"
        )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    if ws.tables:

        for table_name in list(
            ws.tables.keys()
        ):

            del ws.tables[
                table_name
            ]


    if ws.max_row >= 4:

        table = Table(

            displayName="TrainingData",

            ref=(
                f"A3:J{ws.max_row}"
            )
        )


        style = TableStyleInfo(

            name="TableStyleMedium2",

            showFirstColumn=False,

            showLastColumn=False,

            showRowStripes=True,

            showColumnStripes=False
        )


        table.tableStyleInfo = style

        ws.add_table(
            table
        )


    # --------------------------------------------------------
    # REWARD COLOR
    # --------------------------------------------------------

    ws.conditional_formatting.add(

        f"B4:B{ws.max_row}",

        ColorScaleRule(

            start_type="min",
            start_color="F8696B",

            mid_type="percentile",
            mid_value=50,
            mid_color="FFEB84",

            end_type="max",
            end_color="63BE7B"
        )
    )


    wb.save(
        EXCEL_FILE
    )


# ============================================================
# DRAW GRID
# ============================================================

def draw_grid():

    win.fill(
        BACKGROUND
    )


    for x in range(
        GRID_SIZE
    ):

        for y in range(
            GRID_SIZE
        ):

            rect = pygame.Rect(

                x * CELL_SIZE,

                y * CELL_SIZE,

                CELL_SIZE,

                CELL_SIZE
            )


            pygame.draw.rect(

                win,

                (242, 248, 242),

                rect
            )


            pygame.draw.rect(

                win,

                GRID_COLOR,

                rect,

                2,

                border_radius=8
            )


# ============================================================
# DRAW FOOD
# ============================================================

def draw_food(food):

    x = (

        food.position[0]

        *

        CELL_SIZE
    )


    y = (

        food.position[1]

        *

        CELL_SIZE
    )


    center = (

        x + CELL_SIZE // 2,

        y + CELL_SIZE // 2
    )


    pygame.draw.circle(

        win,

        FOOD_RED,

        center,

        CELL_SIZE // 3
    )


    pygame.draw.circle(

        win,

        FOOD_HIGHLIGHT,

        (

            center[0] - 7,

            center[1] - 8
        ),

        4
    )


    # Stem

    pygame.draw.line(

        win,

        STEM_BROWN,

        (

            center[0],

            center[1] - 18
        ),

        (

            center[0] + 4,

            center[1] - 28
        ),

        4
    )


    # Leaf

    pygame.draw.ellipse(

        win,

        LEAF_GREEN,

        (

            center[0] + 3,

            center[1] - 30,

            15,

            8
        )
    )


# ============================================================
# DRAW SNAKE
# ============================================================

def draw_snake(snake):

    for index, pos in enumerate(
        snake.positions
    ):

        x = (

            pos[0]

            *

            CELL_SIZE
        )


        y = (

            pos[1]

            *

            CELL_SIZE
        )


        padding = 4


        rect = pygame.Rect(

            x + padding,

            y + padding,

            CELL_SIZE
            - padding * 2,

            CELL_SIZE
            - padding * 2
        )


        if index == 0:

            # HEAD

            pygame.draw.rect(

                win,

                SNAKE_HEAD,

                rect,

                border_radius=18
            )


            pygame.draw.rect(

                win,

                SNAKE_DARK,

                rect,

                2,

                border_radius=18
            )


            head_x = (
                x + CELL_SIZE // 2
            )

            head_y = (
                y + CELL_SIZE // 2
            )


            dx, dy = (
                snake.direction
            )


            if dx == 1:

                eyes = [

                    (
                        head_x + 9,
                        head_y - 8
                    ),

                    (
                        head_x + 9,
                        head_y + 8
                    )
                ]


            elif dx == -1:

                eyes = [

                    (
                        head_x - 9,
                        head_y - 8
                    ),

                    (
                        head_x - 9,
                        head_y + 8
                    )
                ]


            elif dy == -1:

                eyes = [

                    (
                        head_x - 8,
                        head_y - 9
                    ),

                    (
                        head_x + 8,
                        head_y - 9
                    )
                ]


            else:

                eyes = [

                    (
                        head_x - 8,
                        head_y + 9
                    ),

                    (
                        head_x + 8,
                        head_y + 9
                    )
                ]


            for eye in eyes:

                pygame.draw.circle(

                    win,

                    WHITE,

                    eye,

                    6
                )


                pygame.draw.circle(

                    win,

                    EYE_COLOR,

                    eye,

                    3
                )


        else:

            # BODY

            pygame.draw.rect(

                win,

                SNAKE_BODY,

                rect,

                border_radius=15
            )


            pygame.draw.rect(

                win,

                SNAKE_DARK,

                rect,

                2,

                border_radius=15
            )


# ============================================================
# DRAW INFO
# ============================================================

def draw_info(

    episode,

    total_reward,

    food_eaten,

    epsilon,

    mode

):

    text = (

        f"{mode}   "

        f"Episode: {episode}   "

        f"Reward: {total_reward:.1f}   "

        f"Food: {food_eaten}   "

        f"ε: {epsilon:.3f}"
    )


    surface = small_font.render(

        text,

        True,

        BLACK
    )


    win.blit(

        surface,

        (10, 10)
    )


# ============================================================
# CHOOSE ACTION
# ============================================================

def choose_action(

    snake,

    state_index,

    q_table,

    epsilon_value

):

    valid_actions = get_valid_actions(
        snake
    )


    # No valid actions

    if not valid_actions:

        return 0


    # --------------------------------------------------------
    # EXPLORATION
    # --------------------------------------------------------

    if random.random() < epsilon_value:

        return random.choice(
            valid_actions
        )


    # --------------------------------------------------------
    # EXPLOITATION
    # --------------------------------------------------------

    q_values = q_table[
        state_index
    ]


    # Only consider valid actions

    best_action = valid_actions[0]

    best_value = q_values[
        best_action
    ]


    for action_index in valid_actions[1:]:

        value = q_values[
            action_index
        ]


        if value > best_value:

            best_value = value

            best_action = action_index


    return best_action


# ============================================================
# INITIALIZE
# ============================================================

wb, ws = setup_excel()

last_episode = get_last_episode(
    ws
)

reward_history = load_reward_history(
    ws
)

q_table = load_q_table()


# ============================================================
# TRAINING / TESTING SETUP
# ============================================================

if TRAINING_MODE:

    epsilon = max(

        epsilon_min,

        epsilon
        *
        (
            epsilon_decay
            **
            last_episode
        )
    )


    start_episode = (
        last_episode + 1
    )


    end_episode = (

        start_episode

        +

        num_episodes

        -

        1
    )


    mode_name = "TRAINING"


else:

    # No exploration during testing

    epsilon = 0.0

    start_episode = (
        last_episode + 1
    )

    end_episode = start_episode

    mode_name = "TEST"


# ============================================================
# START INFORMATION
# ============================================================

print(
    "\n======================================"
)

print(
    "       SNAKE Q-LEARNING AI V2"
)

print(
    "======================================"
)

print(
    f"Mode              : {mode_name}"
)

print(
    f"Previous episodes : {last_episode}"
)

print(
    f"Q-table shape     : {q_table.shape}"
)

print(
    f"Q-table states    : {STATE_SIZE}"
)

print(
    f"Starting epsilon  : {epsilon:.4f}"
)

print(
    f"AI memory         : {Q_TABLE_FILE}"
)

print(
    f"Excel log         : {EXCEL_FILE}"
)

print(
    "======================================\n"
)


# ============================================================
# MAIN EPISODE LOOP
# ============================================================

running = True


for episode in range(
    start_episode,
    end_episode + 1
):

    if not running:

        break


    snake = Snake()

    food = Food()

    food.randomize(
        snake
    )


    done = False

    total_reward = 0

    steps = 0

    food_eaten = 0


    # ========================================================
    # EPISODE LOOP
    # ========================================================

    while (

        not done

        and running

        and steps < MAX_STEPS

    ):

        clock.tick(
            FPS
        )


        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

                save_q_table(
                    q_table
                )

                wb.save(
                    EXCEL_FILE
                )

                break


        if not running:

            break


        steps += 1


        # ----------------------------------------------------
        # OLD DISTANCE
        # ----------------------------------------------------

        head_x, head_y = (
            snake.positions[0]
        )

        food_x, food_y = (
            food.position
        )


        old_distance = (

            abs(
                head_x - food_x
            )

            +

            abs(
                head_y - food_y
            )
        )


        # ----------------------------------------------------
        # CURRENT STATE
        # ----------------------------------------------------

        state = get_state(

            snake,

            food
        )


        state_index = state_to_index(
            state
        )


        # ----------------------------------------------------
        # ACTION
        # ----------------------------------------------------

        action_idx = choose_action(

            snake,

            state_index,

            q_table,

            epsilon
        )


        action = ACTIONS[
            action_idx
        ]


        # ----------------------------------------------------
        # MOVE
        # ----------------------------------------------------

        snake.move(
            action
        )


        # ----------------------------------------------------
        # CHECK COLLISION
        # ----------------------------------------------------

        collision = snake.collision()


        # ----------------------------------------------------
        # FOOD CHECK
        # ----------------------------------------------------

        ate_food = (

            not collision

            and

            snake.positions[0]
            == food.position
        )


        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        reward = get_reward(

            snake,

            food,

            old_distance
        )


        total_reward += reward


        # ----------------------------------------------------
        # NEXT STATE
        # ----------------------------------------------------

        if collision:

            target = reward

            done = True


        elif ate_food:

            # Food is terminal only for this transition,
            # but the episode continues.

            snake.grow()

            food_eaten += 1

            food.randomize(
                snake
            )


            next_state = get_state(

                snake,

                food
            )


            next_state_index = (
                state_to_index(
                    next_state
                )
            )


            future_q = np.max(

                q_table[
                    next_state_index
                ]
            )


            target = (

                reward

                +

                gamma
                *

                future_q
            )


        else:

            next_state = get_state(

                snake,

                food
            )


            next_state_index = (
                state_to_index(
                    next_state
                )
            )


            future_q = np.max(

                q_table[
                    next_state_index
                ]
            )


            target = (

                reward

                +

                gamma
                *

                future_q
            )


        # ----------------------------------------------------
        # Q-LEARNING UPDATE
        # ----------------------------------------------------

        old_q = q_table[

            state_index,

            action_idx
        ]


        new_q = (

            old_q

            +

            alpha

            *

            (
                target
                -
                old_q
            )
        )


        q_table[

            state_index,

            action_idx

        ] = new_q


        # ----------------------------------------------------
        # DRAW
        # ----------------------------------------------------

        draw_grid()

        draw_food(
            food
        )

        draw_snake(
            snake
        )

        draw_info(

            episode,

            total_reward,

            food_eaten,

            epsilon,

            mode_name
        )


        pygame.display.update()


    # ========================================================
    # EPISODE FINISHED
    # ========================================================

    if not running:

        break


    if steps >= MAX_STEPS:

        status = "Max Steps"

    else:

        status = "Collision"


    # ========================================================
    # TRAINING ONLY
    # ========================================================

    if TRAINING_MODE:

        reward_history.append(
            total_reward
        )


        average_reward = (

            sum(
                reward_history
            )

            /

            len(
                reward_history
            )
        )


        # ----------------------------------------------------
        # EPSILON DECAY
        # ----------------------------------------------------

        epsilon = max(

            epsilon_min,

            epsilon
            *
            epsilon_decay
        )


        # ----------------------------------------------------
        # SAVE MEMORY
        # ----------------------------------------------------

        save_q_table(
            q_table
        )


        # ----------------------------------------------------
        # SAVE EXCEL
        # ----------------------------------------------------

        update_excel(

            wb=wb,

            ws=ws,

            episode=episode,

            total_reward=total_reward,

            steps=steps,

            food_eaten=food_eaten,

            snake_length=len(
                snake.positions
            ),

            alpha=alpha,

            gamma=gamma,

            epsilon=epsilon,

            average_reward=average_reward,

            status=status
        )


        print(

            f"Episode {episode:05d} | "

            f"Reward: "
            f"{total_reward:8.1f} | "

            f"Steps: "
            f"{steps:4d} | "

            f"Food: "
            f"{food_eaten:3d} | "

            f"Length: "
            f"{len(snake.positions):3d} | "

            f"Epsilon: "
            f"{epsilon:.3f} | "

            f"Memory saved"
        )


    else:

        # ----------------------------------------------------
        # TEST MODE
        # ----------------------------------------------------

        print(

            f"TEST | "

            f"Reward: "
            f"{total_reward:.1f} | "

            f"Steps: "
            f"{steps} | "

            f"Food: "
            f"{food_eaten} | "

            f"Length: "
            f"{len(snake.positions)}"
        )


# ============================================================
# FINAL SAVE
# ============================================================

save_q_table(
    q_table
)

wb.save(
    EXCEL_FILE
)

pygame.quit()


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n======================================"
)

print(
    "        PROGRAM FINISHED"
)

print(
    "======================================"
)

print(
    f"AI memory saved : {Q_TABLE_FILE}"
)

print(
    f"Excel saved     : {EXCEL_FILE}"
)

print(
    f"Total episodes  : {last_episode + num_episodes}"
)

print(
    f"Final epsilon    : {epsilon:.4f}"
)

print(
    "======================================"
)