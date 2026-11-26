"""CPU contract check using the actual verl-agent collector and scripted responses."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from threading import Thread

import numpy as np
from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer
from verl import DataProto
from agent_system.environments import make_envs
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from medical.calculation import FIELDS


def exercise(manager, config, tokenizer, actions, expected=1):
    class ScriptedActor:
        world_size = 1

        def generate_sequences(self, batch):
            response = tokenizer.encode(next(actions), add_special_tokens=False)
            tokens = torch.tensor([response]*len(batch))
            return DataProto.from_dict(tensors={"responses": tokens,
                                                "input_ids": torch.cat((batch.batch["input_ids"], tokens), dim=1)})

    prompts = np.empty(2, dtype=object)
    prompts[:] = [[{"role":"user","content":""}]]*2
    data = DataProto.from_dict(tensors={"input_ids":torch.zeros(2,1,dtype=torch.long)},
                               non_tensors={"raw_prompt":prompts, "data_source":np.array(["text"]*2)})
    rows, returns, lengths, scores, _, _ = TrajectoryCollector(config,tokenizer).vanilla_multi_turn_loop(
        data, ScriptedActor(), manager)
    assert list(scores["success_rate"]) == [expected]*2 and list(returns) == [expected]*2
    return lengths.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixtures", "upstream", "tokenizer"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    class MockSearch(BaseHTTPRequestHandler):
        def do_GET(self):
            self.server.searches += 1
            self.send_response(200)
            self.send_header("Content-Type","application/fhir+json")
            self.end_headers()
            self.wfile.write(b'{"resourceType":"Bundle","total":0,"entry":[]}')
        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1",0),MockSearch) as server:
        server.searches = 0
        Thread(target=server.serve_forever,daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}/fhir/"
        config = OmegaConf.create(dict(data=dict(train_batch_size=1,val_batch_size=2,
                                       max_prompt_length=8192,truncation="error"),
                                       env=dict(env_name="medical/evidence", rollout=dict(n=2),max_steps=8,
                                       medical=dict(fixtures=str(args.fixtures),upstream=str(args.upstream),
                                                    fhir_base=base))))
        train, _ = make_envs(config)
        w, key = train.cases[0]
        actions = [cmd for field in FIELDS[w.calculator]
                   for cmd in (f"QUERY {field}",f"BIND {field} {key['good'][field]}")]
        a = exercise(train,config,tokenizer,iter(actions+["CALCULATE","SUBMIT"]))
        config.env.env_name = "medical/fhir"
        _, val = make_envs(config)
        # Use two fictitious official task1-shaped cases, with a known answer.
        val.cases = [dict(id="task1_fixture",context="",instruction="Return the test label.",
                          sol=["example"],eval_MRN="demo")]*2
        b = exercise(val,config,tokenizer,iter([f"GET {base}Patient?identifier=demo",'FINISH(["example"])']))
        assert server.searches == 2
        config.env.medical.gated = True
        _, gated = make_envs(config)
        gated.cases = [dict(id="task5_fixture", context="", instruction="Review the fixture.",
                            eval_MRN="demo")]*2
        # Each reference scorer performs its own GET; none of those reads
        # may satisfy the agent's gate on an empty submission.
        exercise(gated,config,tokenizer,iter(["FINISH([])"]),expected=0)
        exercise(gated,config,tokenizer,iter([f"GET {base}Observation?patient=demo&code=MG",
                                              "FINISH([])"]))
        assert server.searches == 8  # 4 agent GETs + 4 reference GETs
        server.shutdown()
        print("Evidence episode lengths", a, "FHIR fixture episode lengths", b, "HTTP GETs including scorer", server.searches)


if __name__ == "__main__":
    main()
