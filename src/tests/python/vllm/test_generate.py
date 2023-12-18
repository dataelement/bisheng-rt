#!/usr/bin/python3
# flake8: noqa

import json
import sys
import threading
import time
import unittest

import requests
import sseclient
import test_util as tu

sys.path.append('../common')




class GenerateEndpointTest(tu.TestResultCollector):
    def setUp(self):
        self._model_name = 'Qwen-1_8B-Chat'

    def _get_infer_url(self, model_name, route):
        return f'http://localhost:9001/v2.1/models/{model_name}/{route}'

    def generate_stream(self, model_name, inputs, stream=False):
        headers = {'Accept': 'text/event-stream'}
        url = self._get_infer_url(model_name, 'generate_stream')
        # stream=True used to indicate response can be iterated over, which
        # should be the common setting for generate_stream.
        # For correctness test case, stream=False so that we can re-examine
        # the response content.
        return requests.post(
            url,
            data=inputs if isinstance(inputs, str) else json.dumps(inputs),
            headers=headers,
            stream=stream,
        )

    def generate(self, model_name, inputs):
        url = self._get_infer_url(model_name, 'generate')
        return requests.post(
            url, data=inputs if isinstance(inputs, str) else json.dumps(inputs)
        )

    def generate_expect_failure(self, model_name, inputs, msg):
        url = self._get_infer_url(model_name, 'generate')
        r = requests.post(
            url, data=inputs if isinstance(inputs, str) else json.dumps(inputs)
        )
        try:
            r.raise_for_status()
            self.assertTrue(False, f'Expected failure, success for {inputs}')
        except requests.exceptions.HTTPError as e:
            self.assertIn(msg, r.json()['error'])

    def generate_stream_expect_failure(self, model_name, inputs, msg):
        r = self.generate_stream(model_name, inputs)
        try:
            r.raise_for_status()
            self.assertTrue(False, f'Expected failure, success for {inputs}')
        except requests.exceptions.HTTPError as e:
            self.assertIn(msg, r.json()['error'])

    def check_sse_responses(self, res):
        # Validate SSE format
        self.assertIn('Content-Type', res.headers)
        self.assertIn('text/event-stream', res.headers['Content-Type'])

        # SSE format (data: []) is hard to parse, use helper library for simplicity
        client = sseclient.SSEClient(res)
        res_count = 0
        for event in client.events():
            # Parse event data, join events into a single response
            data = json.loads(event.data)
            print('sse data', data)
            res_count += 1

        # Make sure there is no message in the wrong form
        for remaining in client._read():
            self.assertTrue(
                remaining.startswith(b'data:'),
                f'SSE response not formed properly, got: {remaining}',
            )
            self.assertTrue(
                remaining.endswith(b'\n\n'),
                f'SSE response not formed properly, got: {remaining}',
            )

    def test_generate(self):
        payload = {
          'model': 'Qwen-1.8B-Chat',
          'messages': [
            {'role': 'user', 'content': 'hello'}
           ]
        }

        r = self.generate(self._model_name, payload)
        r.raise_for_status()

        self.assertIn('Content-Type', r.headers)
        self.assertIn('application/json', r.headers['Content-Type'])
        data = r.json()
        self.assertIn('choices', data)

    def test_generate_stream(self):
        payload = {
          'model': 'Qwen-1.8B-Chat',
          'messages': [
            {'role': 'user', 'content': 'hello'}
           ],
          'stream': True,
        }

        r = self.generate_stream(self._model_name, payload)
        self.check_sse_responses(r)

    def test_close_connection_during_streaming(self):
        # verify the responses are streamed as soon as it is generated
        payload = {
          'model': 'Qwen-1.8B-Chat',
          'messages': [
            {'role': 'user', 'content': '以香山为主题，写一篇游记'}
           ],
          'stream': True,
        }

        res = self.generate_stream(self._model_name, payload, stream=True)
        # close connection while the responses are being generated
        res.close()
        # check server healthiness
        health_url = 'http://localhost:9001/v2/health/live'
        requests.get(health_url).raise_for_status()

    def test_race_condition(self):
        # In Triton HTTP frontend, the HTTP response is sent in a different
        # thread than Triton response complete thread, both programs have shared
        # access to the same object, so this test is sending sufficient load to
        # the endpoint, in attempt to expose race condition if any  .
        payload1 = {
          'model': 'Qwen-1.8B-Chat',
          'messages': [
            {'role': 'user', 'content': '以香山为主题，写一篇散文'}
           ],
          'stream': True,
        }

        payload2 = {
          'model': 'Qwen-1.8B-Chat',
          'messages': [
            {'role': 'user', 'content': '以狼牙山为主题，写一篇散文'}
           ],
          'stream': True,
        }

        threads = []

        def thread_func(model_name, inputs):
            self.generate_stream(model_name, inputs).raise_for_status()

        for _ in range(50):
            threads.append(
                threading.Thread(target=thread_func, args=((self._model_name, payload1)))
            )
            threads.append(
                threading.Thread(target=thread_func, args=((self._model_name, payload2)))
            )

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


if __name__ == '__main__':
    unittest.main()
