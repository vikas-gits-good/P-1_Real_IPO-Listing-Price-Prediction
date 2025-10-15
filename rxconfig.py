import reflex as rx

config = rx.Config(
    app_name="Frontend",
    app_module_import="Frontend.test4",  # "Frontend.test_template" #
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
    telemetry_enabled=False,
    # loglevel=rx.config.LogLevel.WARNING,
)
