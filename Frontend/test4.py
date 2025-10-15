import reflex as rx
import pandas as pd

from src.Logging.logger_pred import logging
from src.Exception.exception import CustomException
from src.Pipeline.training_pipeline import TrainIPOPrediction

df = pd.read_csv("src/Data/InitialData/ipo_scrn_gmp_EQ.csv")
cols_list = [
    "IPO_company_name",
    "IPO_face_value",
    "IPO_issue_price",
    "IPO_lot_size",
    "IPO_issue_size",
    "IPO_Broker_apply",
    "IPO_Member_apply",
    "IPO_day1_qib",
    "IPO_day1_nii",
    "IPO_day1_rtl",
    "IPO_day2_qib",
    "IPO_day2_nii",
    "IPO_day2_rtl",
]
ipo_data = df[cols_list]


class State(rx.State):
    latest_data = ipo_data.values.tolist()[:200]
    archive_data = ipo_data.values.tolist()[200:]

    @rx.event
    def retrain_model(self):
        try:
            logging.info(f"{'User Training':-^{60}}")
            TrainIPOPrediction().train()

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    @rx.event
    def update_data(self):
        try:
            logging.info(f"{'User Updating':-^{60}}")
            pass

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    @rx.event
    def predict_data(self):
        try:
            logging.info(f"{'User Predicting':-^{60}}")
            pass

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)


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
            data=State.latest_data,
            columns=list(ipo_data.columns),
            search=True,
            pagination=True,
            sort=True,
        ),
    )


def archive_tab():
    return rx.vstack(
        rx.data_table(
            data=State.archive_data,
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
app.add_page(
    index,
    route="/",
    title="IPO Prediction WebApp",
    description="A web application to analyse latest Mainboard IPOs and predict listing gains",
    on_load=State.predict_data,
)
