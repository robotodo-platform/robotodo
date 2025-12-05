import asyncio


async def main():
    from robotodo.engines.isaac.scene import Scene

    # TODO allow custom
    scene = Scene.load(
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Grid/default_environment.usd",
    )
    scene.viewer.mode = "editing"
    scene.viewer.show()

    # TODO
    await scene._kernel.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
