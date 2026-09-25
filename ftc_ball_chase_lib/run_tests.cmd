@echo off
rem One-click test suite for ftc_ball_chase_lib. Compiles final/ + wrapper/ +
rem tests/ against the staged FTC SDK + Pedro jars and runs every suite:
rem   BallMath         - pure math self-test
rem   WrapperLogicTest - wrapper verb logic (colors, scoring, chains, gather...)
rem   AllTests         - every function + every driver (tracker, controller,
rem                      follower, final BallHunt, MecanumWrangler, PedroWrangler,
rem                      wrapper BallHunt) with mocked motors / scripted vision.
rem On-robot TeleOp harnesses (BallChaseControllerTest) still need the robot.

call "%~dp0compile_check.cmd" -runall
exit /b %errorlevel%