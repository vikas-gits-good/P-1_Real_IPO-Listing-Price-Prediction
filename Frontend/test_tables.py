import reflex as rx
from collections import Counter


class User(rx.Base):
    name: str
    email: str
    gender: str


class State(rx.State):
    users: list[User] = [
        User(name="Danilo", email="danilo@ex.com", gender="Male"),
        User(name="Zahra", email="zahra@ex.com", gender="Female"),
    ]
    users_for_graph: list[dict] = []

    def add_user(self, form_data: dict = None):
        self.users.append(User(**form_data))
        self.transform_data()

    def transform_data(self):
        gend_counts = Counter(user.gender for user in self.users)

        self.users_for_graph = [
            {"name": gender_group, "value": count}
            for gender_group, count in gend_counts.items()
        ]


def show_user(user: User):
    return rx.table.row(
        rx.foreach(
            [user.name, user.email, user.gender], lambda value: rx.table.cell(value)
        ),
        style={"_hover": {"bg": rx.color("gray", 3)}},
        align="center",
    )


def add_customer_button() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.trigger(
            rx.button(
                rx.icon("plus", size=25),
                rx.text("Add user", size="4"),
            ),
        ),
        rx.dialog.content(
            rx.dialog.title("Add new user"),
            rx.dialog.description("Fill the form with new user's information"),
            rx.form(
                rx.flex(
                    rx.input(placeholder="Daniel", name="name", required=True),
                    rx.input(placeholder="daniel@ex.com", name="email", required=False),
                    rx.select(
                        items=["Male", "Female"],
                        placeholder="Male",
                        name="gender",
                        required=True,
                    ),
                    rx.flex(
                        rx.dialog.close(
                            rx.button("Cancel", variant="soft", color_scheme="gray"),
                        ),
                        rx.dialog.close(
                            rx.button("Submit", type="submit"),
                        ),
                        spacing="3",
                        justify="end",
                    ),
                    direction="column",
                    spacing="4",
                ),
                on_submit=State.add_user,
                reset_on_submit=False,
            ),
            max_width="450px",
        ),
    )


def graph():
    return rx.recharts.bar_chart(
        rx.recharts.bar(
            data_key="value",
            stroke=rx.color("accent", 9),
            fill=rx.color("accent", 8),
        ),
        rx.recharts.x_axis(data_key="name"),
        rx.recharts.y_axis(allow_decimals=False),
        data=State.users_for_graph,
        width="100%",
        height=250,
    )


def index() -> rx.Component:
    return rx.vstack(
        add_customer_button(),
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Name"),
                    rx.table.column_header_cell("Email"),
                    rx.table.column_header_cell("Gender"),
                ),
            ),
            rx.table.body(
                rx.foreach(State.users, show_user),
            ),
            variant="surface",
            size="3",
            width="100%",
        ),
        graph(),
        align="center",
        width="100%",
    )


app = rx.App(theme=rx.theme(radius="full", accent_color="grass"))
app.add_page(
    index,
    route="/",
    title="Customer Data App",
    description="A simple app to manage customer data",
    on_load=State.transform_data,
)
app._compile()
