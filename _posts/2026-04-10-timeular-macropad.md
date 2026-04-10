---
layout: post
title: "I turned my Timeular Tracker into a macOS macropad"
date: 2026-04-10 10:00:00-0000
description: "Flipping an 8-sided dice to switch macOS desktops, built end to end with Claude Opus."
tags: [hardware, macOS, Swift, Claude, hack]
categories: [side-projects]
giscus_comments: false
related_posts: false
---

Show and tell time.

<div class="row">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/timeular-tracker.png" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/timeular-menubar.png" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

## What it does

I turned my Timeular Tracker (the 8-sided time-tracking dice) into a physical macropad for macOS. Flip the dice, switch macOS desktops. Side 1 is my main workspace, side 2 is a side project, side 3 is another. No keyboard shortcut to remember, just a satisfying physical flip. Every side can also be bound to launching an app, running a shell command, or opening a URL, all configured from a tiny menu bar popover.

## How it got built

I had the Timeular charged but unused on my desk (I stopped paying the monthly subscription). I also had Claude Opus 4.6, and even though I don't know Swift, Opus does.

1. **Scan.** Asked Claude to find the Timeular over Bluetooth. It picked Python to explore the device.
2. **Prototype.** Read the orientation signal on flip and wired it to shell commands. A working macropad in one script.
3. **Go native.** Rewrote the whole thing in Swift/SwiftUI as a proper macOS menu bar app with CoreBluetooth.
4. **Ship it.** Works.

Source on [GitHub](https://github.com/WindChimeRan/ohmytimular).
