import pygame

class Button:
    def __init__(self, x, y, width, height, text, color=(100, 100, 100), hover_color=(150, 150, 150), text_color=(255, 255, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        # Ensure color values are valid integers
        self.color = tuple(max(0, min(255, int(c))) for c in color)
        self.hover_color = tuple(max(0, min(255, int(c))) for c in hover_color)
        self.text_color = tuple(max(0, min(255, int(c))) for c in text_color)
        self.font = pygame.font.SysFont('arial', 24)  # Default font
        self.is_hovered = False
        self.glow_amount = 0
        self.glow_direction = 1

    def draw(self, screen, custom_font=None):
        # Check if mouse is hovering over button
        mouse_pos = pygame.mouse.get_pos()
        current_color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color

        # Draw button background
        pygame.draw.rect(screen, current_color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)  # Border

        # Use custom font if provided, otherwise use default
        font_to_use = custom_font if custom_font else self.font

        # Draw text
        text_surface = font_to_use.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

    def update_glow(self):
        # Update glow animation
        self.glow_amount += 0.1 * self.glow_direction
        if self.glow_amount >= 1.0:
            self.glow_direction = -1
        elif self.glow_amount <= 0.0:
            self.glow_direction = 1

        # Update glow color based on glow_amount, ensuring integer values
        glow_color = (
            int(min(self.color[0] + 30 * self.glow_amount, 255)),
            int(min(self.color[1] + 30 * self.glow_amount, 255)),
            int(min(self.color[2] + 30 * self.glow_amount, 255))
        )
        self.hover_color = glow_color

    def draw_glow(self, screen):
        # Update glow animation
        self.update_glow()

        # Create button surface with glow
        button_surf = pygame.Surface((self.rect.width + 20, self.rect.height + 20))
        button_surf.fill((0, 0, 0))  # Fill with black for transparency
        button_surf.set_colorkey((0, 0, 0))  # Make black transparent

        # Draw multiple rectangles for glow effect
        glow_alpha = int(50 * self.glow_amount) if self.is_hovered else 30
        for i in range(4, 0, -1):
            glow_rect = pygame.Rect(10-i, 10-i, self.rect.width+i*2, self.rect.height+i*2)
            glow_surface = pygame.Surface((glow_rect.width, glow_rect.height))
            glow_surface.fill((0, 0, 0))
            glow_surface.set_colorkey((0, 0, 0))
            pygame.draw.rect(glow_surface, self.hover_color, glow_surface.get_rect(), border_radius=12)
            glow_surface.set_alpha(glow_alpha)
            button_surf.blit(glow_surface, (glow_rect.x - (10-i), glow_rect.y - (10-i)))

        # Main button rectangle
        main_rect = pygame.Rect(10, 10, self.rect.width, self.rect.height)
        pygame.draw.rect(button_surf, self.hover_color, main_rect, border_radius=12)

        # Add scanlines effect
        scanline_surface = pygame.Surface((self.rect.width, 2))
        scanline_surface.fill((0, 0, 0))
        scanline_surface.set_alpha(50)
        for y in range(0, self.rect.height, 2):
            button_surf.blit(scanline_surface, (10, 10+y))

        # Border with neon effect
        border_surface = pygame.Surface((main_rect.width, main_rect.height))
        border_surface.fill((0, 0, 0))
        border_surface.set_colorkey((0, 0, 0))
        pygame.draw.rect(border_surface, self.hover_color, border_surface.get_rect(), 2, border_radius=12)
        button_surf.blit(border_surface, (10, 10))

        # Draw the button surface
        screen.blit(button_surf, (self.rect.x - 10, self.rect.y - 10))

        # Render text with glow
        text_surface = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)

        # Add text glow
        if self.is_hovered:
            glow_text = text_surface.copy()
            glow_text.set_alpha(50)
            for i in range(3):
                screen.blit(glow_text, (text_rect.x - i, text_rect.y))
                screen.blit(glow_text, (text_rect.x + i, text_rect.y))
                screen.blit(glow_text, (text_rect.x, text_rect.y - i))
                screen.blit(glow_text, (text_rect.x, text_rect.y + i))

        screen.blit(text_surface, text_rect)
