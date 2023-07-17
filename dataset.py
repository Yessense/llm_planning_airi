from ast import Set
from typing import List, Optional
import torch
import json

from torch.utils.data import Dataset

class AlfredDataset(Dataset):
    def __init__(self,
                 split: str,
                 path_to_data_dir: str,
                 available_task_types: Optional[Set[str]] = None,
                 ) -> None:
        super().__init__()

        self._split = split

        if available_task_types is not None:
            self._available_task_types = available_task_types
        else:
            self._available_task_types = set()

        self._available_objects = set()
        self._plans: List[AlfredPlan] = []

        print(f"Loading split: {self._split}...")

        with open(f'{path_to_data_dir}/{self.split}_highlevel.json', 'r') as f:
            data = json.load(f)

            for entry in data:
                # task type
                # 'pick_heat_then_place_in_recep'
                task_type = entry['task_type']
                if available_task_types is not None:
                    if task_type not in available_task_types:
                        continue

                # even than check if task type is possible
                self._available_task_types.add(task_type)

                # May not have subtasks
                list_of_actions = entry['list_of_actions']
                if list_of_actions is None:
                    continue

                # Parse list of actions
                # ['potato', 'SliceObject']
                subtasks = []
                for obj, action in list_of_actions:
                    subtask = Subtask(
                        action=Action(action),
                        arguments=[Arg(obj)],
                        n_arguments=1)  # only one argument
                    subtasks.append(subtask)

                # scene objects
                # ['StoveBurner', 'ButterKnife', 'SoapBottle', 'CounterTop', 'Bowl',
                # 'Microwave', 'Window', 'Fork', 'Egg', 'StoveKnob',
                # 'GarbageCan', 'Plate', 'Mug', 'Sink', ...]
                scene_objects = [SceneObj(name=obj) for obj in entry['objects']]
                self._available_objects.update(scene_objects)

                # Combine all information into a Plan
                plan = AlfredPlan(
                    goal=entry['goal'],
                    task_type=entry['task_type'],
                    subtasks=subtasks,
                    scene_objects=scene_objects)

                self._plans.append(plan)

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, index) -> AlfredPlan:
        return self.plans[index]

    @property
    def plans(self) -> List[AlfredPlan]:
        return self._plans

    @property
    def available_task_types(self) -> Set[str]:
        return self._available_task_types

    @property
    def available_objects(self) -> Set[SceneObj]:
        return self._available_objects

    @property
    def split(self):
        return self._split
