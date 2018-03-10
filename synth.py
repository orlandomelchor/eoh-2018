import pygame
import array

class Note(pygame.mixer.Sound):

    def __init__(self, frequency, volume=.1):
        self.frequency = frequency
        pygame.mixer.Sound.__init__(self, self.build_samples())
        self.set_volume(volume)

    def build_samples(self):
        period = int(round(pygame.mixer.get_init()[0] / self.frequency))
        samples = array.array("h", [0] * period)
        amplitude = 2 ** (abs(pygame.mixer.get_init()[1]) - 1) - 1
        for time in range(period):
            if time < period / 2:
                samples[time] = amplitude
            else:
                samples[time] = -amplitude
        return samples

if __name__ == "__main__":

    # --- init ---

    pygame.mixer.pre_init(44100, -16, 1, 1024)
    pygame.init()

    screen = pygame.display.set_mode((100, 100))

    # --- objects ---
    tones = {
    	pygame.K_q: Note(264), #C3
    	pygame.K_2: Note(281.6), #C#3
    	pygame.K_w: Note(297),#D3
    	pygame.K_3: Note(316.8),#D#3
    	pygame.K_e: Note(330),#E3
    	pygame.K_r: Note(352),#F3
    	pygame.K_5: Note(371.25),#F#3
    	pygame.K_t: Note(396),#G3
    	pygame.K_6: Note(422.4),#G#3
        pygame.K_y: Note(440),#A3
        pygame.K_7: Note(469.0+1.0/3.0),#A#3
        pygame.K_u: Note(495),#B3
        pygame.K_i: Note(528),#C4
    	pygame.K_F1: Note(2*281.6),#C#4
    	pygame.K_F2: Note(2*297),#D4
    	pygame.K_F3: Note(2*316.8),#D#4
    	pygame.K_F4: Note(2*330),#E4
    	pygame.K_F5: Note(2*352),#F4
    	pygame.K_F6: Note(2*371.25),#F#4
    	pygame.K_F7: Note(2*396),#G4
    	pygame.K_F8: Note(2*422.4),#G#4
        pygame.K_F9: Note(2*440),#A4
        pygame.K_F10: Note(2*(469.0+1.0/3.0)),#A#4
        pygame.K_F11: Note(2*495),#B4
        pygame.K_F12: Note(2*528),#C5
        pygame.K_z: Note(264.0/2.0),#C#5
    	pygame.K_s: Note(281.6/2.0),#D5
    	pygame.K_x: Note(297.0/2.0),#D#5
    	pygame.K_d: Note(316.8/2.0),#E5
    	pygame.K_c: Note(330.0/2.0),#F5
    	pygame.K_v: Note(352.0/2.0),#F#5
    	pygame.K_g: Note(371.25/2.0),#G5
    	pygame.K_b: Note(396.0/2.0),#G#5
    	pygame.K_h: Note(422.4/2.0),#A5
        pygame.K_n: Note(440.0/2.0),#A#3
        pygame.K_j: Note((469.0+1.0/3.0)/2.0),#B3
        pygame.K_m: Note(495.0/2.0),#C3
        pygame.K_k: Note(264*3, .1*1/3),
        pygame.K_l: Note(264*5, .1*1/5),
        pygame.K_o: Note(264*(2**(1.0/3.0))),
        pygame.K_p: Note(264*(2**(7.0/12.0)))
    }

    # --- mainloop ---

    running = True

    while running:
        for event in pygame.event.get():

            # closing window
            if event.type == pygame.QUIT:
                running = False

            # pressing key
            elif event.type == pygame.KEYDOWN:
                if event.key in tones:
                    tones[event.key].play(-1)

            # releasing key
            elif event.type == pygame.KEYUP:
                if event.key in tones:
                    tones[event.key].stop()

    # --- end ---

    pygame.quit()