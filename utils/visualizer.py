import pygame
import numpy as np

class TradingVisualizer:
    def __init__(self, window_size=200, width=950, height=600):
        """
        Initializes the enhanced Professional Pygame visualization window.
        """
        pygame.init()
        self.width = width
        self.height = height
        self.window_size = window_size
        
        # Margins to make room for axes and HUD
        self.margin_top = 80
        self.margin_bottom = 50
        self.margin_left = 20
        self.margin_right = 80

        self.graph_width = self.width - self.margin_left - self.margin_right
        self.graph_height = self.height - self.margin_top - self.margin_bottom
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Q-Trade: Algorithmic Trading Terminal")
        
        # --- Modern Light Mode Palette ---
        self.BG_COLOR = (248, 250, 252)       # Very light slate/white
        self.GRID_COLOR = (226, 232, 240)     # Subtle grid lines
        self.AXIS_COLOR = (148, 163, 184)     # Axis lines
        self.LINE_COLOR = (37, 99, 235)       # Professional Finance Blue
        self.BUY_COLOR = (22, 163, 74)        # Clean Green
        self.SELL_COLOR = (220, 38, 38)       # Clean Red
        self.TEXT_COLOR = (51, 65, 85)        # Dark Slate for text
        self.HUD_BG = (255, 255, 255)         # Pure white header
        self.HUD_BORDER = (203, 213, 225)
        
        # Fonts
        self.font_small = pygame.font.SysFont("Segoe UI, Arial", 14)
        self.font = pygame.font.SysFont("Segoe UI, Arial", 20)
        self.large_font = pygame.font.SysFont("Segoe UI, Arial", 28, bold=True)
        
        self.clock = pygame.time.Clock()

    def render(self, price_data, current_step, actions_history, total_profit, inventory, dates=None):
        """
        Renders a single frame of the trading environment.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill(self.BG_COLOR)
        
        # 1. Determine the slice of data to show
        start_idx = max(0, current_step - self.window_size)
        end_idx = current_step
        
        window_prices = price_data[start_idx:end_idx]
        if len(window_prices) < 2:
            return True 
            
        # 2. Dynamic Scaling with Padding
        min_price = np.min(window_prices)
        max_price = np.max(window_prices)
        # Add 5% padding to the top and bottom of the chart
        padding = (max_price - min_price) * 0.05 if max_price != min_price else 1
        min_price -= padding
        max_price += padding
        price_range = max_price - min_price if max_price != min_price else 1
        
        # 3. Draw Grid and Axes Numbers
        # Horizontal Grid Lines & Y-Axis (Prices)
        num_h_lines = 6
        for i in range(num_h_lines):
            y_ratio = i / (num_h_lines - 1)
            y = self.margin_top + y_ratio * self.graph_height
            price_level = max_price - (y_ratio * price_range)
            
            # Grid line
            pygame.draw.line(self.screen, self.GRID_COLOR, (self.margin_left, y), (self.width - self.margin_right, y), 1)
            
            # Price Label
            price_text = self.font_small.render(f"${price_level:.2f}", True, self.TEXT_COLOR)
            self.screen.blit(price_text, (self.width - self.margin_right + 10, y - 8))

        # Vertical Grid Lines & X-Axis (Time Steps)
        num_v_lines = 6
        for i in range(num_v_lines):
            x_ratio = i / (num_v_lines - 1)
            x = self.margin_left + x_ratio * self.graph_width
            step_val = start_idx + int(x_ratio * len(window_prices))
            
            # Grid line
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, self.margin_top), (x, self.height - self.margin_bottom), 1)
            
            # --- REPLACE THE OLD STEP LABEL WITH THIS ---
            if dates and step_val < len(dates):
                label_text = dates[step_val]
            else:
                label_text = f"Step {step_val}"
                
            step_text = self.font_small.render(label_text, True, self.TEXT_COLOR)
            # Draw it slightly shifted to the left to center the date string
            self.screen.blit(step_text, (x - 35, self.height - self.margin_bottom + 15))
        # Draw framing box for the chart area
        pygame.draw.rect(self.screen, self.AXIS_COLOR, (self.margin_left, self.margin_top, self.graph_width, self.graph_height), 1)

        # 4. Calculate Coordinates for the Price Line
        points = []
        for i, price in enumerate(window_prices):
            # Map index to X pixel
            x = self.margin_left + int((i / max(1, len(window_prices) - 1)) * self.graph_width)
            # Map price to Y pixel
            y = self.margin_top + int(self.graph_height - ((price - min_price) / price_range) * self.graph_height)
            points.append((x, y))
            
        # 5. Draw Area Chart Fill and Line
        if len(points) >= 2:
            # Create a transparent surface for the soft blue fill underneath the line
            fill_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            fill_points = [(points[0][0], self.margin_top + self.graph_height)] + points + [(points[-1][0], self.margin_top + self.graph_height)]
            pygame.draw.polygon(fill_surface, (*self.LINE_COLOR, 30), fill_points) # 30 is the alpha transparency
            self.screen.blit(fill_surface, (0, 0))
            
            # Draw the solid price line
            pygame.draw.lines(self.screen, self.LINE_COLOR, False, points, 2)
            
            # Draw current price tracker (pulsing dot and line)
            last_x, last_y = points[-1]
            pygame.draw.circle(self.screen, self.LINE_COLOR, (last_x, last_y), 5)
            pygame.draw.line(self.screen, (191, 219, 254), (self.margin_left, last_y), (self.width - self.margin_right, last_y), 1)
            
            # Current Price Tag on Y-Axis
            current_price = window_prices[-1]
            tag_text = self.font_small.render(f"${current_price:.2f}", True, (255, 255, 255))
            tag_rect = tag_text.get_rect(center=(self.width - self.margin_right / 2 + 5, last_y))
            pygame.draw.rect(self.screen, self.LINE_COLOR, tag_rect.inflate(12, 8), border_radius=4)
            self.screen.blit(tag_text, tag_rect)

        # 6. Draw Buy/Sell Action Markers
        window_actions = actions_history[start_idx:end_idx]
        for i, action in enumerate(window_actions):
            if action in [1, 2]: # Buy or Sell
                x, y = points[i]
                if action == 1: # Buy (Green pointing up)
                    pygame.draw.polygon(self.screen, self.BUY_COLOR, 
                                        [(x, y + 10), (x - 6, y + 20), (x + 6, y + 20)])
                elif action == 2: # Sell (Red pointing down)
                    pygame.draw.polygon(self.screen, self.SELL_COLOR, 
                                        [(x, y - 10), (x - 6, y - 20), (x + 6, y - 20)])

        # 7. Render Professional HUD (Heads Up Display)
        pygame.draw.rect(self.screen, self.HUD_BG, (0, 0, self.width, 65))
        pygame.draw.line(self.screen, self.HUD_BORDER, (0, 65), (self.width, 65), 1)
        
        # Platform Title
        title_surface = self.font.render("Q-Trade AI Desk", True, self.TEXT_COLOR)
        self.screen.blit(title_surface, (20, 20))
        
        # Total Profit Display
        profit_text = f"Total P&L: ${total_profit:.2f}"
        profit_color = self.BUY_COLOR if total_profit >= 0 else self.SELL_COLOR
        prof_surface = self.large_font.render(profit_text, True, profit_color)
        self.screen.blit(prof_surface, (250, 15))
        
        # Inventory Status Pill Badge
        if inventory == 1:
            status_text = "STATUS: LONG (ASSET)"
            status_color = self.LINE_COLOR
        else:
            status_text = "STATUS: FLAT (CASH)"
            status_color = (148, 163, 184)
            
        stat_surface = self.font_small.render(status_text, True, (255, 255, 255))
        stat_rect = stat_surface.get_rect(center=(self.width - 120, 32))
        pygame.draw.rect(self.screen, status_color, stat_rect.inflate(24, 12), border_radius=15)
        self.screen.blit(stat_surface, stat_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)
        
        return True

    def close(self):
        pygame.quit()