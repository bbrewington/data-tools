from manim import *
import os

class ChocolateRicePudding(Scene):
    def construct(self):
        # Set up ingredients
        milk = Text("1 cup milk").scale(0.8)
        cocoa = Text("1/4 cup cocoa powder").scale(0.8)
        rice = Text("1/2 cup rice").scale(0.8)
        sugar = Text("1/4 cup brown sugar").scale(0.8)
        salt = Text("1/4 tsp salt").scale(0.8)
        egg = Text("2 egg yolks").scale(0.8)
        maple = Text("2 tbsp maple syrup").scale(0.8)
        vanilla = Text("1 tsp vanilla extract").scale(0.8)
        whipped_cream = Text("Whipped cream").scale(0.8)
        cocoa_sprinkle = Text("Cocoa powder (for topping)").scale(0.8)

        # Set up steps
        step1_title = Text("Step 1: Cook the Pudding").scale(0.9)
        step1_content1 = Text("1. Combine milk, cocoa powder, rice, brown sugar, and salt in a pot.").scale(0.8)
        step1_content2 = Text("2. Bring to a boil over medium heat, stirring occasionally.").scale(0.8)
        step1_content3 = Text("3. Reduce heat to medium-low and simmer for 20 minutes, stirring often.").scale(0.8)
        step1_group = VGroup(step1_title, step1_content1, step1_content2, step1_content3)

        step2_title = Text("Step 2: Add Yolks and Flavoring").scale(0.9)
        step2_content1 = Text("1. Remove pot from heat.").scale(0.8)
        step2_content2 = Text("2. Add egg yolks, maple syrup, and vanilla extract.").scale(0.8)
        step2_content3 = Text("3. Stir until combined.").scale(0.8)
        step2_content4 = Text("4. Let cool and thicken.").scale(0.8)
        step2_group = VGroup(step2_title, step2_content1, step2_content2, step2_content3, step2_content4)

        step3_title = Text("Step 3: Serve the Pudding").scale(0.9)
        step3_content1 = Text("1. Serve warm or chilled.").scale(0.8)
        step3_content2 = Text("2. Top with whipped cream and a sprinkle of cocoa powder, if desired.").scale(0.8)
        step3_content3 = Text("3. Refrigerate leftovers in an airtight container for up to 2 days.").scale(0.8)
        step3_group = VGroup(step3_title, step3_content1, step3_content2, step3_content3)

        # Animate step 1
        self.play(Write(step1_title))
        self.wait()
        self.play(Write(step1_content1))
        self.play(Write(VGroup(milk, cocoa, rice, sugar, salt)))
        self.wait()
        self.play(Write(step1_content2))
        self.wait()
        self.play(Write(step1_content3))
        self.wait()

        # Animate step 2
        self.play(FadeOut(step1_group))
        self.play(Write(step2_title))
        self.wait()
        self.play(Write(step2_content1))
        self.wait()
        self.play(Write(step2_content2))
        self.play(Write(VGroup(egg, maple, vanilla)))
        self.wait()
        self.play(Write(step2_content3))
        self.wait()
        self.play(Write(step2_content4))
        self.wait()

        # Animate step 3
        self.play(FadeOut(step2_group))
        self.play(Write(step3_title))
        self.wait()
        self.play(Write(step3_content1))
        self.wait()
        self.play(Write(step3_content2))
        self.play(Write(VGroup(whipped_cream, cocoa_sprinkle)))
        self.wait()
        self.play(Write(step3_content3))
        self.wait()

        # End scene
        self.play(FadeOut(step3_group))
        self.wait()