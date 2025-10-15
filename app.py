import reflex as rx
from reflex import html


# Sample data structure for IPO table rows
ipo_data = [
    {
        "company_name": "Company A",
        "ipo_open_date": "2025-01-01",
        "ipo_close_date": "2025-01-10",
        "qibb": 1000,
        "nii": 500,
        "retail_subscription": 300,
        "gmp_value": 50,
        "predicted_listing_gain": 15.2,
        "actual_listing_gain": 17.5,
    },
    # Add more rows up to 20 for example
]

# Replicate more data for demo
ipo_data *= 20
ipo_data = ipo_data[:20]  # Limit to 20 rows


class State(rx.State):
    page_latest = 1
    page_archive = 1
    rows_per_page = 5

    latest_data = ipo_data
    archive_data = ipo_data

    def retrain_model(self):
        # Call your ML model retrain class here
        print("Retraining model...")

    def update_data(self):
        # Code to update database data here
        print("Updating data in DB...")

    def predict_data(self):
        # Re-predict data logic here
        print("Re-predicting data...")

    def change_page_latest(self, page):
        self.page_latest = page

    def change_page_archive(self, page):
        self.page_archive = page

    def paginated_data(self, tab: str):
        if tab == "latest":
            start = (self.page_latest - 1) * self.rows_per_page
            end = start + self.rows_per_page
            return self.latest_data[start:end]
        else:
            start = (self.page_archive - 1) * self.rows_per_page
            end = start + self.rows_per_page
            return self.archive_data[start:end]


def ipo_table(data):
    return rx.table(
        rx.table.thead(
            rx.table.tr(
                rx.table.th("Company Name"),
                rx.table.th("IPO Open Date"),
                rx.table.th("IPO Close Date"),
                rx.table.th("QIBB"),
                rx.table.th("NII"),
                rx.table.th("Retail Subscription"),
                rx.table.th("GMP Value"),
                rx.table.th("Predicted Listing Gain"),
                rx.table.th("Actual Listing Gain"),
            )
        ),
        rx.table.tbody(
            *[
                rx.table.tr(
                    rx.table.td(row["company_name"]),
                    rx.table.td(row["ipo_open_date"]),
                    rx.table.td(row["ipo_close_date"]),
                    rx.table.td(str(row["qibb"])),
                    rx.table.td(str(row["nii"])),
                    rx.table.td(str(row["retail_subscription"])),
                    rx.table.td(str(row["gmp_value"])),
                    rx.table.td(str(row["predicted_listing_gain"])),
                    rx.table.td(str(row["actual_listing_gain"])),
                )
                for row in data
            ]
        ),
        style={
            "width": "100%",
            "border": "1px solid black",
            "borderCollapse": "collapse",
        },
    )


def pagination_controls(tab: str):
    def handle_click(page):
        if tab == "latest":
            return State.change_page_latest(page)
        return State.change_page_archive(page)

    pages = list(range(1, 5))  # Assuming 4 pages max for 20 rows and 5 rows/page
    return rx.hstack(
        *[
            rx.button(
                str(page),
                on_click=lambda page=page: handle_click(page),
                style={"margin": "2px"},
            )
            for page in pages
        ]
    )


def latest_tab():
    return rx.vstack(
        rx.hstack(
            rx.button("Retrain", on_click=State.retrain_model),
            rx.button("Update", on_click=State.update_data),
            rx.button("Predict", on_click=State.predict_data),
        ),
        ipo_table(State.paginated_data("latest")),
        pagination_controls("latest"),
    )


def archive_tab():
    return rx.vstack(
        ipo_table(State.paginated_data("archive")),
        pagination_controls("archive"),
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
