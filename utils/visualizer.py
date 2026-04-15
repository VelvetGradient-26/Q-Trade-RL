import pygame
import numpy as np

class TradingVisualizer:
    def __init__(self, window_size=200, width=800, height=600):
        """
        Initializes the Pygame visualization window.
        :param window_size: How many data points to show on the screen at once (scrolling window).
        """
        pygame.init()
        self.width = width
        self.height = height
        self.window_size = window_size
        
        # Set up the display
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Q-Trade: Algorithmic Swing Trading Agent")
        
        # Colors
        self.BG_COLOR = (20, 20, 20)       # Dark Gray/Black
        self.LINE_COLOR = (100, 200, 255)  # Light Blue
        self.BUY_COLOR = (0, 255, 0)       # Green
        self.SELL_COLOR = (255, 0, 0)      # Red
        self.TEXT_COLOR = (255, 255, 255)  # White
        
        # Fonts
        self.font = pygame.font.SysFont("Arial", 24)
        self.large_font = pygame.font.SysFont("Arial", 32, bold=True)
        
        # Clock for controlling frame rate during rendering
        self.clock = pygame.time.Clock()

    def render(self, price_data, current_step, actions_history, total_profit, inventory):
        """
        Renders a single frame of the trading environment.
        """
        # Handle Pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill(self.BG_COLOR)
        
        # 1. Determine the slice of data to show (scrolling window)
        start_idx = max(0, current_step - self.window_size)
        end_idx = current_step
        
        window_prices = price_data[start_idx:end_idx]
        if len(window_prices) < 2:
            return True # Not enough data to draw a line yet
            
        # 2. Dynamic Scaling
        min_price = np.min(window_prices) - 2 # Padding
        max_price = np.max(window_prices) + 2 # Padding
        price_range = max_price - min_price if max_price != min_price else 1
        
        # Calculate X and Y coordinates for the line graph
        points = []
        for i, price in enumerate(window_prices):
            x = int((i / self.window_size) * self.width)
            # Invert Y because Pygame 0,0 is top-left
            y = int(self.height - ((price - min_price) / price_range) * self.height)
            points.append((x, y))
            
        # 3. Draw the Price Line
        pygame.draw.lines(self.screen, self.LINE_COLOR, False, points, 2)
        
        # 4. Draw Buy/Sell Markers
        # We only look at actions within our current visual window
        window_actions = actions_history[start_idx:end_idx]
        for i, action in enumerate(window_actions):
            if action in [1, 2]: # Buy or Sell
                x, y = points[i]
                
                # Draw Triangles
                if action == 1: # Buy (Green pointing up)
                    pygame.draw.polygon(self.screen, self.BUY_COLOR, 
                                        [(x, y - 10), (x - 8, y + 8), (x + 8, y + 8)])
                elif action == 2: # Sell (Red pointing down)
                    pygame.draw.polygon(self.screen, self.SELL_COLOR, 
                                        [(x, y + 10), (x - 8, y - 8), (x + 8, y - 8)])

        # 5. Render HUD (Heads Up Display)
        # Display Profit
        profit_text = f"Total Profit: ${total_profit:.2f}"
        profit_color = self.BUY_COLOR if total_profit >= 0 else self.SELL_COLOR
        prof_surface = self.large_font.render(profit_text, True, profit_color)
        self.screen.blit(prof_surface, (20, 20))
        
        # Display Inventory Status
        inv_text = "Status: HOLDING ASSET" if inventory == 1 else "Status: HOLDING CASH"
        inv_color = self.LINE_COLOR
        inv_surface = self.font.render(inv_text, True, inv_color)
        self.screen.blit(inv_surface, (20, 60))
        
        # Update display and tick clock
        pygame.display.flip()
        self.clock.tick(30) # Limit to 30 FPS for smooth viewing
        
        return True

    def close(self):
        pygame.quit()