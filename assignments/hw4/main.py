import argparse
from student_code.simple_baseline_experiment_runner import SimpleBaselineExperimentRunner
from student_code.coattention_experiment_runner import CoattentionNetExperimentRunner


if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention'], default='simple')
    parser.add_argument('--train_image_dir', type=str, default='/content/data/train2014')
    parser.add_argument('--train_question_path', type=str, default='/content/data/OpenEnded_mscoco_train2014_questions.json')
    parser.add_argument('--train_annotation_path', type=str, default='/content/data/mscoco_train2014_annotations.json')
    parser.add_argument('--test_image_dir', type=str, default='/content/data/val2014/')
    parser.add_argument('--test_question_path', type=str, default='/content/data/OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--test_annotation_path', type=str, default='/content/data/mscoco_val2014_annotations.json')
    parser.add_argument('--batch_size', type=int, default=125)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_data_loader_workers', type=int, default=2)
    parser.add_argument('--cache_location', type=str, default="")
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--log_validation', action='store_true')
    args = parser.parse_args()

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    elif args.model == "coattention":
        experiment_runner_class = CoattentionNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers,
                                                cache_location=args.cache_location,
                                                lr=args.lr,
                                                log_validation=args.log_validation)
    experiment_runner.train()
