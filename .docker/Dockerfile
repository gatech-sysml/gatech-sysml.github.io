FROM ruby:3.4-bookworm@sha256:246b2dc3f6e40bba3af18503c22997a34dbb27c9f97e198dde6dd727895115c5
WORKDIR /usr/src/app
ENV VIRTUAL_ENV=/opt/cite-venv
ENV PATH="/opt/cite-venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv && rm -rf /var/lib/apt/lists/*
COPY Gemfile Gemfile.lock ./
RUN bundle install
COPY _cite/requirements.txt /tmp/cite-requirements.txt
RUN python3 -m venv "$VIRTUAL_ENV" && pip install --no-cache-dir -r /tmp/cite-requirements.txt 'watchdog[watchmedo]==6.0.0'
COPY .docker/entrypoint.sh /usr/local/bin/site-entrypoint
COPY .docker/preview.yaml /opt/preview.yaml
EXPOSE 4000 35729
ENTRYPOINT ["site-entrypoint"]
CMD ["site"]
