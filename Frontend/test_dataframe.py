import reflex as rx
import pandas as pd

data_dict = {
    "company_name": ["Company A"] * 20,
    "ipo_open_date": ["2025-01-01"] * 20,
    "ipo_close_date": ["2025-01-10"] * 20,
    "qib": [1000] * 20,
    "nii": [500] * 20,
    "rtl": [300] * 20,
    "gmp_value": [50] * 20,
    "predicted_listing_gain": [15.2] * 20,
    "actual_listing_gain": [17.5] * 20,
}
ipo_data = pd.DataFrame(data_dict)


class State(rx.State):
    latest_data = ipo_data
    archive_data = ipo_data

    def retrain_model(self):
        print("Retraining model...")

    def update_data(self):
        print("Updating data in DB...")

    def predict_data(self):
        print("Re-predicting data...")


def latest_tab():
    return rx.vstack(
        rx.hstack(
            rx.button("Retrain", on_click=State.retrain_model),
            rx.button("Update", on_click=State.update_data),
            rx.button("Predict", on_click=State.predict_data),
            spacing="9",
            padding_bottom="10px",
        ),
        rx.data_table(
            data=ipo_data.values.tolist(),
            columns=list(ipo_data.columns),
            search=True,
            pagination=True,
            sort=True,
        ),
    )


def archive_tab():
    return rx.vstack(
        rx.data_table(
            data=ipo_data.values.tolist(),
            columns=list(ipo_data.columns),
            search=True,
            pagination=True,
            sort=True,
        )
    )


def index():
    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger("Latest", value="latest"),
            rx.tabs.trigger("Archive", value="archive"),
        ),
        rx.tabs.content(
            latest_tab(),
            value="latest",
        ),
        rx.tabs.content(
            archive_tab(),
            value="archive",
        ),
        default_value="latest",
    )


app = rx.App(State)
app.add_page(index)
app._compile()
