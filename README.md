# direction_prediction_model

This repository is to showcase my direction prediction model which was utilized in production for active trading purposes and guided decision making for myself as an active portfolio manager. This model was utilized to predict the direction of a given security based on price action data and technical indicators. Price action data was gathered, then technical values were derived. After gathering, the data is fed into a keras model which would provide a continuous value at output. The function here was built to handle securities which did and did not have a saved model. Securities which had not been seen before were sent into an initial modeling phase to generate a model and save it. Afterwards, a saved model is utilized for returning securites.

The second leg of this model was to generate a single 'walk-forward' value. The historical data, price action and technicals, were fed to an extemely simple Keras model for each variable. The single future value for each variable is then saved into a list, which is then fed into the aforementioned Keras model to then make the following session prediction. As mentioned, the prediction here is continuous. The 'classification' aspect of this model was as simple as determining if the previous day's prediction was below or above the current one. After determining direction, the function utilized DB hookups which would store variables related to prediction. This variables were dates, ticker values, predictions, and true outcomes. The storage of data in the DB allowed for ad-hoc querying on performance analytics.

The *cute* part of this project was the connection to a bot built using the [Discord API](https://discord.com/developers/docs/intro). This connection allowed for a user to provide a list of securites to the bot who would then send the function into action and return results for the provided securites.

I featured my sole month of usage on LinkedIn as a daily post about performance as well as market discussion. The below link is the final post in the series, but features links to previous posts as well as the initial post.

LinkedIn Final Post:
https://www.linkedin.com/feed/update/urn:li:activity:6820080050206068736/

**NOTE**:
This repository has excluded live DB hookups and will only work for newly provided tickers. Remnants of DB connections remain, but the initialization of the DB connection has been redacted and causes one logic path of the function to fail. 

* The repository here features development notes in a Jupyter notebook as well as finalized production scripting.
